//! Minimal JSON-Schema normalization for tool arguments.
//!
//! Models that emit XML-style arguments (`<parameter name="count">3</parameter>`)
//! produce raw strings. This module coerces them to the types declared by the
//! OpenAI tool schema. It intentionally does not validate required/enum/format
//! constraints: the gateway stays a best-effort shaper, not a validator.

use std::collections::BTreeMap;

use serde_json::{Map, Number, Value};

use super::Tool;

/// Normalized parameter schemas for all tools in one request.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolSchemas {
    tools: BTreeMap<String, ToolSchema>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct ToolSchema {
    params: BTreeMap<String, JsonParamType>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum JsonParamType {
    String,
    Integer,
    Number,
    Boolean,
    Object,
    Array,
    Null,
    Any,
}

impl ToolSchemas {
    /// Normalize OpenAI-style tool schemas for one request.
    pub fn from_tools(tools: &[Tool]) -> Self {
        let tools = tools
            .iter()
            .map(|tool| (tool.name.clone(), ToolSchema::from_schema(&tool.parameters)))
            .collect();
        Self { tools }
    }

    /// Convert one raw parameter value for `tool_name`.
    ///
    /// `force_string` is set by formats that carry an explicit `string="true"`
    /// attribute, which must win over the schema.
    pub fn convert_param(
        &self,
        tool_name: &str,
        param_name: &str,
        raw: &str,
        force_string: bool,
    ) -> Value {
        if force_string {
            return Value::String(raw.to_string());
        }
        let param_type = self
            .tools
            .get(tool_name)
            .and_then(|tool| tool.params.get(param_name));
        convert_with_type(param_type, raw)
    }
}

impl ToolSchema {
    fn from_schema(parameters: &Value) -> Self {
        let Some(properties) = parameters.get("properties").and_then(Value::as_object) else {
            return Self::default();
        };
        let params = properties
            .iter()
            .filter_map(|(name, schema)| {
                JsonParamType::from_schema(schema).map(|kind| (name.clone(), kind))
            })
            .collect();
        Self { params }
    }
}

impl JsonParamType {
    fn from_schema(schema: &Value) -> Option<Self> {
        let schema = schema.as_object()?;
        if let Some(type_value) = schema.get("type") {
            return match type_value {
                Value::String(name) => Self::from_name(name),
                Value::Array(names) => {
                    let kinds: Vec<Self> = names
                        .iter()
                        .filter_map(Value::as_str)
                        .filter_map(Self::from_name)
                        .collect();
                    if kinds.is_empty() {
                        None
                    } else {
                        // Preserve the first declared type; union coercion is
                        // intentionally conservative.
                        kinds.into_iter().next()
                    }
                }
                _ => None,
            };
        }
        if schema.contains_key("properties") || schema.contains_key("additionalProperties") {
            return Some(Self::Object);
        }
        if schema.contains_key("items") {
            return Some(Self::Array);
        }
        if schema.get("enum").and_then(Value::as_array).is_some() {
            return Some(Self::String);
        }
        None
    }

    fn from_name(name: &str) -> Option<Self> {
        let name = name.trim().to_ascii_lowercase();
        match name.as_str() {
            "string" | "str" | "text" | "varchar" | "char" | "enum" => Some(Self::String),
            "integer" | "int" => Some(Self::Integer),
            "number" | "float" | "double" => Some(Self::Number),
            "boolean" | "bool" => Some(Self::Boolean),
            "object" | "dict" | "map" => Some(Self::Object),
            "array" | "list" | "sequence" => Some(Self::Array),
            "null" => Some(Self::Null),
            _ if name.starts_with("int")
                || name.starts_with("uint")
                || name.starts_with("long")
                || name.starts_with("short") =>
            {
                Some(Self::Integer)
            }
            _ if name.starts_with("num") || name.starts_with("float") => Some(Self::Number),
            _ => Some(Self::Any),
        }
    }
}

fn convert_with_type(param_type: Option<&JsonParamType>, raw: &str) -> Value {
    let Some(param_type) = param_type else {
        // No schema: keep the model's JSON value when it is valid, otherwise
        // return the raw string. This is useful for DeepSeek's `string="false"`
        // parameters even when the request omitted a schema.
        return serde_json::from_str::<Value>(raw)
            .unwrap_or_else(|_| Value::String(raw.to_string()));
    };
    match param_type {
        JsonParamType::String => Value::String(raw.to_string()),
        JsonParamType::Integer => raw
            .parse::<i64>()
            .map(Number::from)
            .map(Value::Number)
            .unwrap_or_else(|_| Value::String(raw.to_string())),
        JsonParamType::Number => {
            if let Ok(number) = serde_json::from_str::<Number>(raw) {
                return Value::Number(number);
            }
            if let Ok(value) = raw.parse::<i64>() {
                return Value::Number(value.into());
            }
            if let Some(number) = raw.parse::<f64>().ok().and_then(Number::from_f64) {
                return Value::Number(number);
            }
            Value::String(raw.to_string())
        }
        JsonParamType::Boolean => match raw.trim().to_ascii_lowercase().as_str() {
            "true" | "1" => Value::Bool(true),
            "false" | "0" => Value::Bool(false),
            _ => Value::String(raw.to_string()),
        },
        JsonParamType::Object | JsonParamType::Array => serde_json::from_str::<Value>(raw)
            .unwrap_or_else(|_| Value::String(raw.to_string())),
        JsonParamType::Null => {
            if raw.eq_ignore_ascii_case("null") {
                Value::Null
            } else {
                Value::String(raw.to_string())
            }
        }
        JsonParamType::Any => serde_json::from_str::<Value>(raw)
            .unwrap_or_else(|_| Value::String(raw.to_string())),
    }
}

/// Build a JSON object string from ordered `(name, value)` pairs.
pub fn arguments_object(pairs: Vec<(String, Value)>) -> Result<String, String> {
    let mut object = Map::with_capacity(pairs.len());
    for (name, value) in pairs {
        object.insert(name, value);
    }
    serde_json::to_string(&Value::Object(object))
        .map_err(|error| format!("failed to serialize arguments: {error}"))
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use super::ToolSchemas;
    use crate::parser::tool::Tool;

    fn tool(name: &str, parameters: Value) -> Tool {
        Tool {
            name: name.to_string(),
            description: None,
            parameters,
            strict: None,
        }
    }

    #[test]
    fn coerces_declared_types() {
        let schemas = ToolSchemas::from_tools(&[tool(
            "f",
            json!({
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                    "ratio": {"type": "number"},
                    "enabled": {"type": "boolean"},
                    "payload": {"type": "object"},
                    "items": {"type": "array"},
                    "text": {"type": "string"}
                }
            }),
        )]);
        assert_eq!(schemas.convert_param("f", "count", "42", false), json!(42));
        assert_eq!(schemas.convert_param("f", "ratio", "2.5", false), json!(2.5));
        assert_eq!(schemas.convert_param("f", "enabled", "true", false), json!(true));
        assert_eq!(
            schemas.convert_param("f", "payload", "{\"x\":1}", false),
            json!({"x": 1})
        );
        assert_eq!(schemas.convert_param("f", "items", "[1,2]", false), json!([1, 2]));
        assert_eq!(schemas.convert_param("f", "text", "42", false), json!("42"));
    }

    #[test]
    fn force_string_wins_and_unknown_schema_falls_back_to_json() {
        let schemas = ToolSchemas::from_tools(&[]);
        assert_eq!(schemas.convert_param("f", "count", "42", true), json!("42"));
        assert_eq!(schemas.convert_param("f", "count", "42", false), json!(42));
        assert_eq!(schemas.convert_param("f", "flag", "true", false), json!(true));
        assert_eq!(schemas.convert_param("f", "text", "hello", false), json!("hello"));
    }
}