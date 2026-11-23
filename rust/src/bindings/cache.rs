//! PyO3 binding for the radix prefix cache.

use std::cell::RefCell;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::cache::radix;

/// Invoke a Python page-ownership hook; a raise is captured for re-throw.
///
/// The hook receives one positional argument: the tuple of page ids.
fn call_hook(
    hook: &Bound<'_, PyAny>,
    raised: &RefCell<Option<PyErr>>,
    pages: &[i64],
) -> Result<(), ()> {
    let py = hook.py();
    let tuple = PyTuple::new_bound(py, pages.iter().map(|page| page.to_object(py)));
    let args = PyTuple::new_bound(py, [tuple]);
    match hook.call1(args) {
        Ok(_) => Ok(()),
        Err(err) => {
            *raised.borrow_mut() = Some(err);
            Err(())
        }
    }
}

fn radix_error(err: radix::RadixErr, raised: &RefCell<Option<PyErr>>) -> PyErr {
    match err {
        radix::RadixErr::Msg(message) => PyValueError::new_err(message),
        radix::RadixErr::Callback => match raised.borrow_mut().take() {
            Some(err) => err,
            None => PyValueError::new_err("radix page hook failed"),
        },
    }
}

/// Python-facing wrapper around the Rust radix prefix cache.
///
/// Page-ownership callbacks are invoked exactly like the Python original; a
/// callback raise propagates the original exception untouched.
#[pyclass]
pub(super) struct PyRadixPrefixCache {
    inner: radix::RadixPrefixCache,
}

#[pymethods]
impl PyRadixPrefixCache {
    #[new]
    fn new(page_size: usize, trust_page_ownership: bool) -> PyResult<Self> {
        match radix::RadixPrefixCache::new(page_size, trust_page_ownership) {
            Ok(inner) => Ok(Self { inner }),
            Err(radix::RadixErr::Msg(message)) => Err(PyValueError::new_err(message)),
            Err(radix::RadixErr::Callback) => {
                Err(PyValueError::new_err("radix page hook failed"))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn match_prefix(
        &mut self,
        token_count: usize,
        pages: Vec<radix::PageKey>,
        namespace: &str,
        reserve_tokens: usize,
        pin: bool,
    ) -> PyResult<(u64, Option<(u64, u64, Vec<i64>)>)> {
        match self.inner.match_prefix(token_count, &pages, namespace, reserve_tokens, pin) {
            Ok(result) => Ok((
                result.eligible_tokens,
                result
                    .handle
                    .map(|handle| (handle.node_id, handle.cached_tokens, handle.page_ids)),
            )),
            Err(radix::RadixErr::Msg(message)) => Err(PyValueError::new_err(message)),
            Err(radix::RadixErr::Callback) => {
                Err(PyValueError::new_err("radix page hook failed"))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (token_count, pages, namespace, page_ids, pin, replace=None, *, retain, release))]
    fn insert_prefix(
        &mut self,
        py: Python<'_>,
        token_count: usize,
        pages: Vec<radix::PageKey>,
        namespace: &str,
        page_ids: Vec<i64>,
        pin: bool,
        replace: Option<(u64, u64)>,
        retain: &Bound<'_, PyAny>,
        release: &Bound<'_, PyAny>,
    ) -> PyResult<(Option<(u64, u64, Vec<i64>)>, u64, u64)> {
        let raised: RefCell<Option<PyErr>> = RefCell::new(None);
        let mut retain_hook = |pages: &[i64]| call_hook(retain, &raised, pages);
        let mut release_hook = |pages: &[i64]| call_hook(release, &raised, pages);
        let _ = py;
        match self.inner.insert_prefix(
            token_count,
            &pages,
            namespace,
            &page_ids,
            pin,
            replace,
            &mut retain_hook,
            &mut release_hook,
        ) {
            Ok(outcome) => Ok((
                outcome
                    .handle
                    .map(|handle| (handle.node_id, handle.cached_tokens, handle.page_ids)),
                outcome.already_cached_pages,
                outcome.inserted_pages,
            )),
            Err(err) => Err(radix_error(err, &raised)),
        }
    }

    fn pin(&mut self, node_id: u64) -> PyResult<()> {
        match self.inner.pin(node_id) {
            Ok(()) => Ok(()),
            Err(radix::RadixErr::Msg(message)) => Err(PyValueError::new_err(message)),
            Err(radix::RadixErr::Callback) => {
                Err(PyValueError::new_err("radix page hook failed"))
            }
        }
    }

    fn unpin(&mut self, node_id: u64) -> PyResult<()> {
        match self.inner.unpin(node_id) {
            Ok(()) => Ok(()),
            Err(radix::RadixErr::Msg(message)) => Err(PyValueError::new_err(message)),
            Err(radix::RadixErr::Callback) => {
                Err(PyValueError::new_err("radix page hook failed"))
            }
        }
    }

    fn evict_pages(
        &mut self,
        _py: Python<'_>,
        target_pages: usize,
        release: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<i64>> {
        let raised: RefCell<Option<PyErr>> = RefCell::new(None);
        let mut release_hook = |pages: &[i64]| call_hook(release, &raised, pages);
        match self.inner.evict_pages(target_pages, &mut release_hook) {
            Ok(pages) => Ok(pages),
            Err(err) => Err(radix_error(err, &raised)),
        }
    }

    fn reset(&mut self, _py: Python<'_>, release: &Bound<'_, PyAny>) -> PyResult<()> {
        let raised: RefCell<Option<PyErr>> = RefCell::new(None);
        let mut release_hook = |pages: &[i64]| call_hook(release, &raised, pages);
        match self.inner.reset(&mut release_hook) {
            Ok(()) => Ok(()),
            Err(err) => Err(radix_error(err, &raised)),
        }
    }

    fn root_namespaces(&self) -> Vec<(String, u64)> {
        self.inner.root_namespaces()
    }

    fn cached_page_ids(&self) -> Vec<i64> {
        self.inner.cached_page_ids()
    }

    fn assert_integrity(&self) -> PyResult<()> {
        match self.inner.assert_integrity() {
            Ok(()) => Ok(()),
            Err(radix::RadixErr::Msg(message)) => Err(PyValueError::new_err(message)),
            Err(radix::RadixErr::Callback) => {
                Err(PyValueError::new_err("radix page hook failed"))
            }
        }
    }

    fn stats(&self) -> Vec<u64> {
        let stats = self.inner.stats();
        vec![
            stats.nodes,
            stats.cached_pages,
            stats.flattened_page_refs,
            stats.pinned_pages,
            stats.evictable_pages,
            stats.lookups,
            stats.hits,
            stats.misses,
            stats.lookup_pages,
            stats.hit_pages,
            stats.insertions,
            stats.inserted_pages,
            stats.evictions,
            stats.evicted_pages,
        ]
    }
}
