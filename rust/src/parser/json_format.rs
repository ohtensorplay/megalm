//! Best-effort `response_format` output validation.

/// Best-effort validator for OpenAI `response_format` outputs.
///
/// The gateway nudges the model with a prompt instruction and checks the
/// generated text here. This is structural validation, not constrained
/// decoding; true grammar enforcement belongs in the engine.
#[derive(Debug)]
pub struct JsonFormatValidator {
    schema: Option<serde_json::Value>,
}

impl JsonFormatValidator {
    pub fn new(schema: Option<serde_json::Value>) -> Self {
        Self { schema }
    }

    pub fn validate(&self, text: &str) -> Result<(), String> {
        let value: serde_json::Value = serde_json::from_str(text.trim())
            .map_err(|error| format!("output is not valid JSON: {error}"))?;
        match &self.schema {
            None => {
                if value.is_object() {
                    Ok(())
                } else {
                    Err("json_object response_format requires a JSON object".into())
                }
            }
            Some(schema) => validate_json_schema(&value, schema),
        }
    }
}

fn json_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(number) if number.is_i64() || number.is_u64() => "integer",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

fn json_type_matches(value: &serde_json::Value, expected: &str) -> bool {
    match expected {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.is_i64() || value.is_u64(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => true,
    }
}

/// Validate a JSON subset: type, enum, required and typed properties.
/// Unknown schema keywords are ignored so permissive schemas stay usable.
fn validate_json_schema(
    value: &serde_json::Value,
    schema: &serde_json::Value,
) -> Result<(), String> {
    let Some(object) = schema.as_object() else {
        return Ok(());
    };
    if let Some(expected) = object.get("type").and_then(|v| v.as_str()) {
        if !json_type_matches(value, expected) {
            return Err(format!(
                "expected {expected}, got {}",
                json_type_name(value)
            ));
        }
    }
    if let Some(enum_values) = object.get("enum").and_then(|v| v.as_array()) {
        if !enum_values.contains(value) {
            return Err("value is not one of the schema enum values".into());
        }
    }
    if let Some(value_object) = value.as_object() {
        if let Some(required) = object.get("required").and_then(|v| v.as_array()) {
            for key in required.iter().filter_map(|v| v.as_str()) {
                if !value_object.contains_key(key) {
                    return Err(format!("missing required property '{key}'"));
                }
            }
        }
        if let Some(properties) = object.get("properties").and_then(|v| v.as_object()) {
            for (key, child_schema) in properties {
                if let Some(child) = value_object.get(key) {
                    validate_json_schema(child, child_schema)
                        .map_err(|error| format!("property '{key}': {error}"))?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod json_format_tests {
    use super::JsonFormatValidator;

    #[test]
    fn json_object_requires_an_object() {
        let validator = JsonFormatValidator::new(None);
        assert!(validator.validate(r#"{"a":1}"#).is_ok());
        assert!(validator.validate("[1,2]").is_err());
        assert!(validator.validate("not json").is_err());
    }

    #[test]
    fn json_schema_checks_type_required_and_properties() {
        let schema: serde_json::Value = serde_json::from_str(
            r#"{"type":"object","required":["name"],"properties":{"name":{"type":"string"},"age":{"type":"integer"}}}"#,
        )
        .unwrap();
        let validator = JsonFormatValidator::new(Some(schema));
        assert!(validator.validate(r#"{"name":"Ada","age":36}"#).is_ok());
        assert!(validator.validate(r#"{"age":36}"#).is_err());
        assert!(validator.validate(r#"{"name":"Ada","age":"x"}"#).is_err());
    }

    #[test]
    fn enum_is_checked() {
        let schema: serde_json::Value = serde_json::from_str(
            r#"{"type":"object","properties":{"color":{"enum":["red","blue"]}}}"#,
        )
        .unwrap();
        let validator = JsonFormatValidator::new(Some(schema));
        assert!(validator.validate(r#"{"color":"red"}"#).is_ok());
        assert!(validator.validate(r#"{"color":"green"}"#).is_err());
    }
}

