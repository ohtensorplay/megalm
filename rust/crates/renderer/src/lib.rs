//! Trusted local chat-template rendering.
//!
//! The environment mirrors Hugging Face's `apply_chat_template` Jinja setup:
//! no autoescape, strict undefined, `trim_blocks` / `lstrip_blocks`, Python
//! methods on primitive values, and the `tojson`, `raise_exception` and
//! `strftime_now` helpers. Templates are always loaded from the model
//! directory; no remote template is fetched.

pub mod error;

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::path::Path;

use minijinja::value::{Kwargs, Value as JValue};
use minijinja::{AutoEscape, Environment, Error as MError, ErrorKind, UndefinedBehavior};
use minijinja_contrib::pycompat::unknown_method_callback;
use serde_json::Value;

pub use error::{RenderError, Result};

const TEMPLATE_NAME: &str = "chat";
const TOOL_TEMPLATE_NAME: &str = "chat::tool_use";

/// Inputs the caller supplies to the chat template.
#[derive(Debug, Clone)]
pub struct RenderContext {
    pub messages: Vec<Value>,
    pub tools: Option<Vec<Value>>,
    pub open_thinking: bool,
    pub bos_token: String,
    pub eos_token: String,
    pub pad_token: String,
    pub unk_token: String,
    /// Additional template variables (for example `reasoning_effort`).
    /// Reserved chat variables cannot be overwritten through this map.
    pub extras: Option<Value>,
}

/// A compiled chat template for one model directory.
pub struct ChatRenderer {
    env: Environment<'static>,
    source: String,
    tool_template: bool,
}

impl ChatRenderer {
    /// Load the template from `model_path`, using the same resolution order as
    /// Hugging Face: a single `chat_template.jinja`, the new
    /// `chat_templates/<name>.jinja` layout, then
    /// `tokenizer_config.json["chat_template"]` and
    /// `chat_template.json["chat_template"]`. Either config field may hold a
    /// named-template object; `default` (or `tool_use`) selects a template.
    pub fn new(model_path: &Path) -> Result<Self> {
        let jinja_path = model_path.join("chat_template.jinja");
        let templates = if jinja_path.is_file() {
            let text = std::fs::read_to_string(&jinja_path).map_err(|source| RenderError::Io {
                path: jinja_path.display().to_string(),
                source,
            })?;
            Some(named_templates(text))
        } else {
            let dir = model_path.join("chat_templates");
            if dir.is_dir() {
                let templates = read_template_dir(&dir)?;
                if !templates.is_empty() {
                    Some(templates)
                } else {
                    None
                }
            } else {
                None
            }
        };
        let templates = match templates {
            Some(templates) => templates,
            None => {
                let missing = || {
                    RenderError::Message(format!(
                        "model has no supported chat template under {}",
                        model_path.display()
                    ))
                };
                let config = read_object(&model_path.join("tokenizer_config.json"))?;
                match config.get("chat_template") {
                    Some(value) => templates_from_value(value).ok_or_else(missing)?,
                    None => {
                        let payload = read_object(&model_path.join("chat_template.json"))?;
                        payload
                            .get("chat_template")
                            .and_then(templates_from_value)
                            .ok_or_else(missing)?
                    }
                }
            }
        };
        Self::from_templates(templates)
    }

    /// Build a renderer from an already-resolved template source.
    pub fn from_source(source: String) -> Result<Self> {
        Self::from_templates(named_templates(source))
    }

    fn from_templates(mut templates: HashMap<String, String>) -> Result<Self> {
        if templates.is_empty() {
            return Err(RenderError::Message(
                "chat template object contains no string template".into(),
            ));
        }
        let default = templates
            .remove("default")
            .or_else(|| templates.remove("chat"))
            .unwrap_or_else(|| {
                let key = templates
                    .keys()
                    .min()
                    .cloned()
                    .expect("templates is not empty");
                templates.remove(&key).expect("key came from templates")
            });
        let tool_source = templates.remove("tool_use");
        let mut env = build_environment();
        env.add_template_owned(TEMPLATE_NAME, strip_generation_tags(&default))?;
        let tool_template = if let Some(source) = tool_source {
            env.add_template_owned(TOOL_TEMPLATE_NAME, strip_generation_tags(&source))?;
            true
        } else {
            false
        };
        Ok(Self {
            env,
            source: default,
            tool_template,
        })
    }

    /// The default template source (with generation markers stripped), used
    /// by tool-parser detection.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Render the assistant prompt for one request.
    pub fn render(&self, ctx: &RenderContext) -> Result<String> {
        let mut vars = serde_json::Map::new();
        vars.insert("messages".into(), Value::Array(ctx.messages.clone()));
        vars.insert(
            "tools".into(),
            match &ctx.tools {
                Some(tools) if !tools.is_empty() => Value::Array(tools.clone()),
                _ => Value::Null,
            },
        );
        vars.insert("add_generation_prompt".into(), Value::Bool(true));
        vars.insert("open_thinking".into(), Value::Bool(ctx.open_thinking));
        vars.insert("enable_thinking".into(), Value::Bool(ctx.open_thinking));
        vars.insert("add_vision_id".into(), Value::Bool(false));
        vars.insert("add_audio_id".into(), Value::Bool(false));
        vars.insert("documents".into(), Value::Null);
        vars.insert("bos_token".into(), Value::String(ctx.bos_token.clone()));
        vars.insert("eos_token".into(), Value::String(ctx.eos_token.clone()));
        vars.insert("pad_token".into(), Value::String(ctx.pad_token.clone()));
        vars.insert("unk_token".into(), Value::String(ctx.unk_token.clone()));
        if let Some(extras) = &ctx.extras {
            if let Value::Object(map) = extras {
                for (key, value) in map {
                    if !vars.contains_key(key) {
                        vars.insert(key.clone(), value.clone());
                    }
                }
            }
        }
        let name =
            if self.tool_template && ctx.tools.as_ref().is_some_and(|tools| !tools.is_empty()) {
                TOOL_TEMPLATE_NAME
            } else {
                TEMPLATE_NAME
            };
        let template = self.env.get_template(name)?;
        Ok(template.render(Value::Object(vars))?)
    }
}

fn build_environment() -> Environment<'static> {
    let mut env = Environment::new();
    env.set_auto_escape_callback(|_| AutoEscape::None);
    env.set_undefined_behavior(UndefinedBehavior::Strict);
    env.set_keep_trailing_newline(true);
    // HF's `apply_chat_template` compiles with these two flags; real model
    // templates (Qwen, TensorMind, ...) rely on them for whitespace control.
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    // Python string/dict/list methods such as `content.startswith(...)` are
    // part of the HF template dialect but unknown to minijinja's core.
    env.set_unknown_method_callback(unknown_method_callback);
    env.add_filter("tojson", tojson);
    env.add_global("raise_exception", JValue::from_function(raise_exception));
    env.add_global("strftime_now", JValue::from_function(strftime_now));
    env
}

fn read_object(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path).map_err(|source| RenderError::Io {
        path: path.display().to_string(),
        source,
    })?;
    serde_json::from_str(&text).map_err(|source| RenderError::Json {
        path: path.display().to_string(),
        source,
    })
}

fn read_template_dir(dir: &Path) -> Result<HashMap<String, String>> {
    let mut templates = HashMap::new();
    for entry in std::fs::read_dir(dir).map_err(|source| RenderError::Io {
        path: dir.display().to_string(),
        source,
    })? {
        let entry = entry.map_err(|source| RenderError::Io {
            path: dir.display().to_string(),
            source,
        })?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("jinja") || !path.is_file() {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        let text = std::fs::read_to_string(&path).map_err(|source| RenderError::Io {
            path: path.display().to_string(),
            source,
        })?;
        templates.insert(stem.to_string(), text);
    }
    Ok(templates)
}

fn named_templates(source: String) -> HashMap<String, String> {
    HashMap::from([("default".to_string(), source)])
}

fn templates_from_value(value: &Value) -> Option<HashMap<String, String>> {
    match value {
        Value::String(text) => Some(named_templates(text.clone())),
        Value::Object(map) => {
            let templates: HashMap<_, _> = map
                .iter()
                .filter_map(|(name, value)| {
                    value.as_str().map(|text| (name.clone(), text.to_string()))
                })
                .collect();
            (!templates.is_empty()).then_some(templates)
        }
        _ => None,
    }
}

/// HF's `{% generation %}` / `{% endgeneration %}` blocks are unknown to
/// Jinja. Strip only the block markers and keep their inner content. Whitespace
/// control markers (`{%- ... -%}`) and extra spacing are accepted.
fn strip_generation_tags(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(start) = rest.find("{%") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let Some(end) = after.find("%}") else {
            out.push_str(&rest[start..]);
            return out;
        };
        let tag = after[..end]
            .trim()
            .trim_start_matches('-')
            .trim_end_matches('-')
            .trim();
        if tag != "generation" && tag != "endgeneration" {
            out.push_str("{%");
            out.push_str(&after[..end]);
            out.push_str("%}");
        }
        rest = &after[end + 2..];
    }
    out.push_str(rest);
    out
}

/// vLLM / Hugging Face extended `tojson` filter. Supports `sort_keys`,
/// `separators`, `ensure_ascii` and `indent`, and defaults to Python's
/// `json.dumps` separators (`", "` / `": "`) rather than compact JSON.
fn tojson(value: JValue, kwargs: Kwargs) -> std::result::Result<String, MError> {
    let sort_keys: Option<bool> = kwargs.get("sort_keys")?;
    let separators: Option<Vec<String>> = kwargs.get("separators")?;
    let ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    // Python's `json.dumps` accepts an int or a literal string as `indent`.
    let indent: Option<String> = match kwargs.get::<Option<JValue>>("indent")? {
        Some(value) if value.as_i64().is_some() => {
            let spaces = value.as_i64().unwrap_or(0).clamp(0, 64) as usize;
            Some(" ".repeat(spaces))
        }
        Some(value) => value.as_str().map(str::to_string),
        None => None,
    };
    // Unknown kwargs are intentionally ignored, matching Python's `**kwargs`.

    let value: Value = serde_json::to_value(&value)
        .map_err(|err| MError::new(ErrorKind::InvalidOperation, err.to_string()))?;
    let value = if sort_keys.unwrap_or(false) {
        sort_json(value)
    } else {
        value
    };
    let compact = separators
        .filter(|parts| parts.len() == 2)
        .map(|parts| (parts[0].clone(), parts[1].clone()));
    let (item_sep, key_sep) = compact.unwrap_or_else(|| {
        if indent.is_some() {
            (",".to_string(), ": ".to_string())
        } else {
            (", ".to_string(), ": ".to_string())
        }
    });
    let mut buffer = String::new();
    write_json(
        &value,
        &mut buffer,
        0,
        indent.as_deref(),
        &item_sep,
        &key_sep,
        ensure_ascii.unwrap_or(false),
    );
    Ok(buffer)
}

fn raise_exception(message: String) -> std::result::Result<(), MError> {
    Err(MError::new(ErrorKind::InvalidOperation, message))
}

/// HF template helper: current local time formatted with `strftime`.
fn strftime_now(format: String) -> std::result::Result<String, MError> {
    let format = CString::new(format)
        .map_err(|err| MError::new(ErrorKind::InvalidOperation, err.to_string()))?;
    let mut now: libc::time_t = 0;
    let mut tm = unsafe { std::mem::zeroed::<libc::tm>() };
    let mut buffer = [0_i8; 256];
    unsafe {
        libc::time(&mut now);
        if libc::localtime_r(&now, &mut tm).is_null() {
            return Err(MError::new(
                ErrorKind::InvalidOperation,
                "cannot determine local time",
            ));
        }
        let written = libc::strftime(buffer.as_mut_ptr(), buffer.len(), format.as_ptr(), &tm);
        if written == 0 {
            return Ok(String::new());
        }
        Ok(CStr::from_ptr(buffer.as_ptr())
            .to_string_lossy()
            .into_owned())
    }
}

fn sort_json(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<String> = map.keys().cloned().collect();
            keys.sort();
            let mut sorted = serde_json::Map::new();
            for key in keys {
                if let Some(item) = map.get(&key) {
                    sorted.insert(key, sort_json(item.clone()));
                }
            }
            Value::Object(sorted)
        }
        Value::Array(items) => Value::Array(items.into_iter().map(sort_json).collect()),
        other => other,
    }
}

#[allow(clippy::too_many_arguments)]
fn write_json(
    value: &Value,
    out: &mut String,
    depth: usize,
    indent: Option<&str>,
    item_sep: &str,
    key_sep: &str,
    ensure_ascii: bool,
) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => out.push_str(&number.to_string()),
        Value::String(text) => write_json_string(text, out, ensure_ascii),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(item_sep);
                }
                write_indent(out, depth + 1, indent);
                write_json(
                    item,
                    out,
                    depth + 1,
                    indent,
                    item_sep,
                    key_sep,
                    ensure_ascii,
                );
            }
            if !items.is_empty() {
                write_indent(out, depth, indent);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    out.push_str(item_sep);
                }
                write_indent(out, depth + 1, indent);
                write_json_string(key, out, ensure_ascii);
                out.push_str(key_sep);
                write_json(
                    item,
                    out,
                    depth + 1,
                    indent,
                    item_sep,
                    key_sep,
                    ensure_ascii,
                );
            }
            if !map.is_empty() {
                write_indent(out, depth, indent);
            }
            out.push('}');
        }
    }
}

fn write_indent(out: &mut String, depth: usize, indent: Option<&str>) {
    if let Some(unit) = indent {
        out.push('\n');
        for _ in 0..depth {
            out.push_str(unit);
        }
    }
}

fn write_json_string(text: &str, out: &mut String, ensure_ascii: bool) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if c.is_ascii() && !c.is_control() => out.push(c),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c if ensure_ascii => {
                let code = c as u32;
                if code <= 0xffff {
                    out.push_str(&format!("\\u{code:04x}"));
                } else {
                    let offset = code - 0x10000;
                    let high = 0xd800 + (offset >> 10);
                    let low = 0xdc00 + (offset & 0x3ff);
                    out.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}
