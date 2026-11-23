//! PyO3 binding for `response_format` validation.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::parser::JsonFormatValidator;

#[pyclass]
pub(super) struct PyJsonFormatValidator {
    inner: JsonFormatValidator,
}

#[pymethods]
impl PyJsonFormatValidator {
    #[new]
    #[pyo3(signature = (schema_json=None))]
    fn new(schema_json: Option<String>) -> PyResult<Self> {
        let schema = match schema_json {
            Some(text) => Some(
                serde_json::from_str(&text)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
            None => None,
        };
        Ok(Self {
            inner: JsonFormatValidator::new(schema),
        })
    }

    fn validate(&self, text: &str) -> PyResult<()> {
        self.inner.validate(text).map_err(PyValueError::new_err)
    }
}

