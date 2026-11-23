//! Page-aligned compressed radix cache — Rust port of
//! `megalm.runtime.radix_cache.RadixPrefixCache`.
//!
//! Semantics are preserved one-to-one (namespaces, pin accounting, LRU with
//! lazy invalidation, page-ownership validation, integrity contract); the
//! Python side becomes a thin facade. Page-ownership callbacks propagate
//! their original Python exceptions via [`RadixErr::Callback`].

use std::collections::{BinaryHeap, HashMap};
use std::cmp::Reverse;

pub type PageKey = Vec<i64>;

/// Evicted-node lookup failure; each call site reports its id for debugging.
fn evicted() -> RadixErr {
    RadixErr::Msg(format!(
        "radix handle references an evicted node\n{}",
        std::backtrace::Backtrace::force_capture()
    ))
}

/// Cache-level error: a contract message, or a callback that already raised.
#[derive(Debug)]
pub enum RadixErr {
    Msg(String),
    Callback,
}

pub type RResult<T> = Result<T, RadixErr>;

pub struct RadixNode {
    pub node_id: u64,
    pub token_pages: Vec<PageKey>,
    pub page_ids: Vec<i64>,
    pub full_page_ids: Vec<i64>,
    pub parent: Option<u64>,
    pub children: HashMap<PageKey, u64>,
    pub direct_pins: i64,
    pub subtree_pins: i64,
    pub last_access: u64,
    pub lru_version: u64,
}

impl RadixNode {
    fn new_root(node_id: u64) -> Self {
        Self {
            node_id,
            token_pages: Vec::new(),
            page_ids: Vec::new(),
            full_page_ids: Vec::new(),
            parent: None,
            children: HashMap::new(),
            direct_pins: 0,
            subtree_pins: 0,
            last_access: 0,
            lru_version: 0,
        }
    }

    fn is_root(&self) -> bool {
        self.parent.is_none()
    }

    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

/// Handle payload returned to the Python facade.
pub struct RadixHandle {
    pub node_id: u64,
    pub cached_tokens: u64,
    pub page_ids: Vec<i64>,
}

pub struct RadixMatchResult {
    pub handle: Option<RadixHandle>,
    pub eligible_tokens: u64,
}

pub struct RadixInsertOutcome {
    pub handle: Option<RadixHandle>,
    pub already_cached_pages: u64,
    pub inserted_pages: u64,
}

pub struct RadixPrefixCache {
    page_size: usize,
    trust_page_ownership: bool,
    next_node_id: u64,
    clock: u64,
    root: u64,
    roots: HashMap<String, u64>,
    root_namespaces: HashMap<u64, String>,
    nodes: HashMap<u64, RadixNode>,
    /// Min-heap of `(last_access, lru_version, node_id)`.
    lru_heap: BinaryHeap<Reverse<(u64, u64, u64)>>,
    cached_pages: i64,
    flattened_page_refs: i64,
    pinned_pages: i64,
    lookups: u64,
    hits: u64,
    misses: u64,
    lookup_pages: u64,
    hit_pages: u64,
    insertions: u64,
    inserted_pages: u64,
    evictions: u64,
    evicted_pages: u64,
}

fn msg<T>(text: &str) -> RResult<T> {
    Err(RadixErr::Msg(text.to_string()))
}

impl RadixPrefixCache {
    pub fn new(page_size: usize, trust_page_ownership: bool) -> RResult<Self> {
        if page_size == 0 {
            return msg("radix page size must be positive");
        }
        let mut nodes = HashMap::new();
        nodes.insert(0u64, RadixNode::new_root(0));
        Ok(Self {
            page_size,
            trust_page_ownership,
            next_node_id: 1,
            clock: 0,
            root: 0,
            roots: HashMap::from([("" .to_string(), 0u64)]),
            root_namespaces: HashMap::from([(0u64, "".to_string())]),
            nodes,
            lru_heap: BinaryHeap::new(),
            cached_pages: 0,
            flattened_page_refs: 0,
            pinned_pages: 0,
            lookups: 0,
            hits: 0,
            misses: 0,
            lookup_pages: 0,
            hit_pages: 0,
            insertions: 0,
            inserted_pages: 0,
            evictions: 0,
            evicted_pages: 0,
        })
    }

    // Public library API: parity accessors for the Python cache's read-only
    // properties. The PyO3 facade exposes them through `stats()`, so the
    // compiler sees no direct caller inside this crate.
    #[allow(dead_code)]
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    #[allow(dead_code)]
    pub fn cached_pages(&self) -> i64 {
        self.cached_pages
    }

    #[allow(dead_code)]
    pub fn pinned_pages(&self) -> i64 {
        self.pinned_pages
    }

    #[allow(dead_code)]
    pub fn evictable_pages(&self) -> i64 {
        self.cached_pages - self.pinned_pages
    }

    pub fn stats(&self) -> RadixStats {
        RadixStats {
            nodes: (self.nodes.len() - self.roots.len()) as u64,
            cached_pages: self.cached_pages as u64,
            flattened_page_refs: self.flattened_page_refs as u64,
            pinned_pages: self.pinned_pages as u64,
            evictable_pages: (self.cached_pages - self.pinned_pages) as u64,
            lookups: self.lookups,
            hits: self.hits,
            misses: self.misses,
            lookup_pages: self.lookup_pages,
            hit_pages: self.hit_pages,
            insertions: self.insertions,
            inserted_pages: self.inserted_pages,
            evictions: self.evictions,
            evicted_pages: self.evicted_pages,
        }
    }

    pub fn cached_page_ids(&self) -> Vec<i64> {
        let mut pages = Vec::new();
        for node in self.nodes.values() {
            if !node.is_root() {
                pages.extend_from_slice(&node.page_ids);
            }
        }
        pages
    }

    /// Namespace → root-node view, mirroring the Python-era `_roots` map.
    pub fn root_namespaces(&self) -> Vec<(String, u64)> {
        self.roots
            .iter()
            .map(|(namespace, root)| (namespace.clone(), *root))
            .collect()
    }

    fn node(&self, node_id: u64) -> RResult<&RadixNode> {
        self.nodes.get(&node_id).ok_or_else(evicted)
    }

    fn node_mut(&mut self, node_id: u64) -> RResult<&mut RadixNode> {
        self.nodes
            .get_mut(&node_id)
            .ok_or_else(|| RadixErr::Msg("radix handle references an evicted node".into()))
    }

    fn find_root(&self, node_id: u64) -> RResult<u64> {
        let mut current = node_id;
        loop {
            let node = self.node(current)?;
            match node.parent {
                None => return Ok(current),
                Some(parent) => current = parent,
            }
        }
    }

    fn make_handle(&self, node_id: u64) -> RResult<RadixHandle> {
        let node = self.node(node_id)?;
        let cached_tokens = (node.full_page_ids.len() * self.page_size) as u64;
        Ok(RadixHandle {
            node_id,
            cached_tokens,
            page_ids: node.full_page_ids.clone(),
        })
    }

    /// Longest common prefix of an edge and the query window, binary-searched.
    fn common_prefix_pages(
        edge: &[PageKey],
        query: &[PageKey],
        query_offset: usize,
        query_limit: usize,
    ) -> usize {
        let limit = edge.len().min(query_limit.saturating_sub(query_offset));
        if limit == 0 {
            return 0;
        }
        if edge[..limit] == query[query_offset..query_offset + limit] {
            return limit;
        }
        let mut low = 0usize;
        let mut high = limit;
        while low < high {
            let middle = (low + high + 1) / 2;
            if edge[..middle] == query[query_offset..query_offset + middle] {
                low = middle;
            } else {
                high = middle - 1;
            }
        }
        low
    }

    fn touch(&mut self, node_id: u64) {
        self.clock += 1;
        let node = match self.nodes.get_mut(&node_id) {
            Some(node) => node,
            None => return,
        };
        node.last_access = self.clock;
        node.lru_version += 1;
        if node.is_leaf() && !node.is_root() {
            let entry = (node.last_access, node.lru_version, node.node_id);
            self.lru_heap.push(Reverse(entry));
        }
    }

    fn touch_path(&mut self, node_id: u64) {
        let mut current = node_id;
        loop {
            let node = match self.nodes.get(&current) {
                Some(node) => node,
                None => break,
            };
            if node.is_root() {
                break;
            }
            let parent = node.parent.expect("non-root has a parent");
            self.touch(current);
            current = parent;
        }
        self.maybe_compact_lru();
    }

    fn maybe_compact_lru(&mut self) {
        let threshold = 64usize.max(4 * self.nodes.len());
        if self.lru_heap.len() <= threshold {
            return;
        }
        self.lru_heap = self
            .nodes
            .values()
            .filter(|node| node.is_leaf() && !node.is_root())
            .map(|node| Reverse((node.last_access, node.lru_version, node.node_id)))
            .collect();
    }

    /// Pop the coldest evictable leaf; pinned leaves are parked for re-push.
    fn pop_evictable_leaf(
        &mut self,
        pinned_entries: &mut Vec<(u64, u64, u64)>,
    ) -> Option<u64> {
        while let Some(Reverse((last_access, version, node_id))) = self.lru_heap.pop() {
            let node = match self.nodes.get(&node_id) {
                Some(node) => node,
                None => continue,
            };
            if node.last_access != last_access
                || node.lru_version != version
                || !node.is_leaf()
            {
                continue;
            }
            if node.subtree_pins != 0 {
                pinned_entries.push((last_access, version, node_id));
                continue;
            }
            return Some(node_id);
        }
        None
    }

    fn new_node(
        &mut self,
        parent: u64,
        token_pages: Vec<PageKey>,
        page_ids: Vec<i64>,
    ) -> RResult<u64> {
        if token_pages.is_empty() || token_pages.len() != page_ids.len() {
            return msg("a radix edge requires matching token/KV pages");
        }
        let parent_node = self
            .nodes
            .get(&parent)
            .ok_or_else(evicted)?;
        let full_page_ids = {
            let mut full = parent_node.full_page_ids.clone();
            full.extend_from_slice(&page_ids);
            full
        };
        let edge_key = token_pages[0].clone();
        let node_id = self.next_node_id;
        self.next_node_id += 1;
        let node = RadixNode {
            node_id,
            token_pages,
            page_ids,
            full_page_ids,
            parent: Some(parent),
            children: HashMap::new(),
            direct_pins: 0,
            subtree_pins: 0,
            last_access: 0,
            lru_version: 0,
        };
        let page_count = node.page_ids.len() as i64;
        let flattened_delta = node.full_page_ids.len() as i64;
        let parent_children = &mut self
            .nodes
            .get_mut(&parent)
            .expect("parent checked above")
            .children;
        if parent_children.contains_key(&edge_key) {
            return msg("radix parent already contains this edge");
        }
        parent_children.insert(edge_key, node_id);
        self.nodes.insert(node_id, node);
        self.cached_pages += page_count;
        self.flattened_page_refs += flattened_delta;
        Ok(node_id)
    }

    fn split_node(&mut self, node_id: u64, prefix_pages: usize) -> RResult<u64> {
        let node = self.node(node_id)?;
        if prefix_pages == 0 || prefix_pages >= node.page_ids.len() {
            return msg("radix split must occur inside an edge");
        }
        let parent = match node.parent {
            Some(parent) => parent,
            None => return msg("cannot split the radix root"),
        };
        let old_key = node.token_pages[0].clone();
        let (prefix_token_pages, node_token_pages) = {
            let mut split = node.token_pages.clone();
            let head = split.drain(..prefix_pages).collect::<Vec<_>>();
            (head, split)
        };
        let (prefix_page_ids, node_page_ids) = {
            let mut split = node.page_ids.clone();
            let head = split.drain(..prefix_pages).collect::<Vec<_>>();
            (head, split)
        };
        let parent_full = self.node(parent)?.full_page_ids.clone();
        let mut prefix_full = parent_full;
        prefix_full.extend_from_slice(&prefix_page_ids);
        let prefix_node_id = self.next_node_id;
        self.next_node_id += 1;
        let (subtree_pins, last_access, node_lru_version) = {
            let node = self.node(node_id)?;
            (node.subtree_pins, node.last_access, node.lru_version)
        };
        let prefix_node = RadixNode {
            node_id: prefix_node_id,
            token_pages: prefix_token_pages,
            page_ids: prefix_page_ids,
            full_page_ids: prefix_full,
            parent: Some(parent),
            children: HashMap::new(),
            direct_pins: 0,
            subtree_pins,
            last_access,
            lru_version: node_lru_version,
        };
        let flattened_delta = prefix_node.full_page_ids.len() as i64;
        let node_first_key = node_token_pages[0].clone();
        {
            let parent_node = self
                .nodes
                .get_mut(&parent)
                .ok_or_else(evicted)?;
            // The edge key already belongs to this parent; the prefix node
            // takes it over verbatim (Python assigns without a conflict check).
            parent_node.children.insert(old_key, prefix_node_id);
        }
        {
            let node = self
                .nodes
                .get_mut(&node_id)
                .ok_or_else(evicted)?;
            node.token_pages = node_token_pages;
            node.page_ids = node_page_ids;
            node.parent = Some(prefix_node_id);
        }
        let mut prefix_node = prefix_node;
        prefix_node.children.insert(node_first_key, node_id);
        self.nodes.insert(prefix_node_id, prefix_node);
        self.flattened_page_refs += flattened_delta;
        Ok(prefix_node_id)
    }

    fn create_namespace_root(&mut self, namespace: &str) -> RResult<u64> {
        if self.roots.contains_key(namespace) {
            return msg("radix namespace root already exists");
        }
        let node_id = self.next_node_id;
        self.next_node_id += 1;
        self.nodes.insert(node_id, RadixNode::new_root(node_id));
        self.roots.insert(namespace.to_string(), node_id);
        self.root_namespaces.insert(node_id, namespace.to_string());
        Ok(node_id)
    }

    fn drop_namespace_root(&mut self, root_id: u64) -> RResult<()> {
        if root_id == self.root {
            return Ok(());
        }
        {
            let root = self.node(root_id)?;
            if !root.is_root() || !root.is_leaf() || !root.page_ids.is_empty() {
                return msg("cannot drop a non-empty radix namespace");
            }
        }
        let namespace = match self.root_namespaces.get(&root_id) {
            Some(namespace) => namespace.clone(),
            None => return msg("radix namespace root is not registered"),
        };
        if self.roots.get(&namespace).copied() != Some(root_id) {
            return msg("radix namespace root is not registered");
        }
        self.roots.remove(&namespace);
        self.root_namespaces.remove(&root_id);
        self.nodes.remove(&root_id);
        Ok(())
    }

    fn pin_node(&mut self, node_id: u64) -> RResult<()> {
        let node = self
            .nodes
            .get_mut(&node_id)
            .ok_or_else(|| RadixErr::Msg("radix handle references an evicted node".into()))?;
        node.direct_pins += 1;
        let mut current = node_id;
        loop {
            let (parent, page_count, cold) = {
                let node = self.node(current)?;
                if node.is_root() {
                    return Ok(());
                }
                (
                    node.parent.expect("non-root has a parent"),
                    node.page_ids.len() as i64,
                    node.subtree_pins == 0,
                )
            };
            if cold {
                self.pinned_pages += page_count;
            }
            let node = self
                .nodes
                .get_mut(&current)
                .ok_or_else(|| RadixErr::Msg("radix handle references an evicted node".into()))?;
            node.subtree_pins += 1;
            current = parent;
        }
    }

    fn unpin_node(&mut self, node_id: u64) -> RResult<()> {
        {
            let node = self
                .nodes
                .get_mut(&node_id)
                .ok_or_else(|| RadixErr::Msg("radix handle references an evicted node".into()))?;
            if node.direct_pins <= 0 {
                return msg("radix handle is not pinned");
            }
            node.direct_pins -= 1;
        }
        let mut pinned_pages = self.pinned_pages;
        let mut current = node_id;
        loop {
            let (parent, page_count, was_pinned) = {
                let node = self.node(current)?;
                if node.is_root() {
                    break;
                }
                (
                    node.parent.expect("non-root has a parent"),
                    node.page_ids.len() as i64,
                    node.subtree_pins > 0,
                )
            };
            if !was_pinned {
                return msg("radix pin accounting underflow");
            }
            let node = self
                .nodes
                .get_mut(&current)
                .ok_or_else(|| RadixErr::Msg("radix handle references an evicted node".into()))?;
            node.subtree_pins -= 1;
            if node.subtree_pins == 0 {
                pinned_pages -= page_count;
            }
            current = parent;
        }
        self.pinned_pages = pinned_pages;
        Ok(())
    }

    fn validate_new_page_ids(&self, parent: u64, suffix_pages: &[i64]) -> RResult<()> {
        let mut suffix_set = std::collections::HashSet::new();
        for page in suffix_pages {
            if !suffix_set.insert(*page) {
                return msg("one prefix cannot contain duplicate KV pages");
            }
        }
        let mut current = parent;
        loop {
            let node = self.node(current)?;
            if node.is_root() {
                return Ok(());
            }
            if node.page_ids.iter().any(|page| suffix_set.contains(page)) {
                return msg("new radix suffix aliases a cached prefix KV page");
            }
            current = node.parent.expect("non-root has a parent");
        }
    }

    fn validate_handle_key(
        &self,
        node_id: u64,
        query_pages: &[PageKey],
        page_count: usize,
    ) -> RResult<()> {
        let mut path: Vec<u64> = Vec::new();
        let mut current = node_id;
        loop {
            let node = self.node(current)?;
            if node.is_root() {
                break;
            }
            path.push(current);
            current = node.parent.expect("non-root has a parent");
        }
        let mut offset = 0usize;
        for path_node_id in path.iter().rev() {
            let path_node = self.node(*path_node_id)?;
            let next_offset = offset + path_node.token_pages.len();
            if next_offset > query_pages.len()
                || path_node.token_pages[..] != query_pages[offset..next_offset]
            {
                return msg("replacement radix handle belongs to another prefix");
            }
            offset = next_offset;
        }
        if offset != page_count {
            return msg("replacement radix handle key length is inconsistent");
        }
        Ok(())
    }
}

pub struct RadixStats {
    pub nodes: u64,
    pub cached_pages: u64,
    pub flattened_page_refs: u64,
    pub pinned_pages: u64,
    pub evictable_pages: u64,
    pub lookups: u64,
    pub hits: u64,
    pub misses: u64,
    pub lookup_pages: u64,
    pub hit_pages: u64,
    pub insertions: u64,
    pub inserted_pages: u64,
    pub evictions: u64,
    pub evicted_pages: u64,
}

/// Page-ownership callback contract: `Err(())` means the Python callable has
/// already raised and the original exception must propagate untouched.
type PageHook<'a> = dyn FnMut(&[i64]) -> Result<(), ()> + 'a;

impl RadixPrefixCache {
    pub fn match_prefix(
        &mut self,
        token_count: usize,
        pages: &[PageKey],
        namespace: &str,
        reserve_tokens: usize,
        pin: bool,
    ) -> RResult<RadixMatchResult> {
        let eligible_tokens = token_count.saturating_sub(reserve_tokens);
        let eligible_pages = eligible_tokens / self.page_size;
        let root = self.roots.get(namespace).copied();
        self.lookups += 1;
        self.lookup_pages += eligible_pages as u64;
        if eligible_pages == 0 {
            self.misses += 1;
            return Ok(RadixMatchResult {
                handle: None,
                eligible_tokens: 0,
            });
        }
        let root = match root {
            Some(root) => root,
            None => {
                self.misses += 1;
                return Ok(RadixMatchResult {
                    handle: None,
                    eligible_tokens: (eligible_pages * self.page_size) as u64,
                });
            }
        };

        let mut node = root;
        let mut page_offset = 0usize;
        while page_offset < eligible_pages {
            let child_id = {
                let node_ref = self.node(node)?;
                match node_ref.children.get(&pages[page_offset]) {
                    Some(child_id) => *child_id,
                    None => break,
                }
            };
            let common = {
                let child = self.node(child_id)?;
                Self::common_prefix_pages(
                    &child.token_pages,
                    pages,
                    page_offset,
                    eligible_pages,
                )
            };
            if common == 0 {
                return msg("radix child key does not match its edge");
            }
            if common < self.node(child_id)?.page_ids.len() {
                node = self.split_node(child_id, common)?;
                page_offset += common;
                break;
            }
            node = child_id;
            page_offset += common;
        }

        if page_offset == 0 {
            self.touch_path(node);
            self.misses += 1;
            return Ok(RadixMatchResult {
                handle: None,
                eligible_tokens: (eligible_pages * self.page_size) as u64,
            });
        }
        self.hits += 1;
        self.hit_pages += page_offset as u64;
        let handle = self.make_handle(node)?;
        if handle.cached_tokens != (page_offset * self.page_size) as u64 {
            return msg("matched handle length is inconsistent");
        }
        if pin {
            self.pin_node(node)?;
        }
        self.touch_path(node);
        Ok(RadixMatchResult {
            handle: Some(handle),
            eligible_tokens: (eligible_pages * self.page_size) as u64,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn insert_prefix(
        &mut self,
        token_count: usize,
        pages: &[PageKey],
        namespace: &str,
        page_ids: &[i64],
        pin: bool,
        replace: Option<(u64, u64)>,
        retain: &mut PageHook,
        release: &mut PageHook,
    ) -> RResult<RadixInsertOutcome> {
        let cacheable_pages = token_count / self.page_size;
        if page_ids.len() < cacheable_pages {
            return msg("prefix insertion has fewer physical pages than token pages");
        }
        let physical_pages = &page_ids[..cacheable_pages];
        self.insertions += 1;
        if cacheable_pages == 0 {
            return Ok(RadixInsertOutcome {
                handle: None,
                already_cached_pages: 0,
                inserted_pages: 0,
            });
        }

        let created_root = !self.roots.contains_key(namespace);
        if created_root && replace.is_some() {
            return msg("replacement radix handle belongs to another namespace");
        }
        let root_id = match self.roots.get(namespace).copied() {
            Some(root_id) => root_id,
            None => self.create_namespace_root(namespace)?,
        };

        let result = self.insert_walk(
            root_id,
            cacheable_pages,
            pages,
            physical_pages,
            replace,
            retain,
            release,
        );
        let (node_id, inserted_pages) = match result {
            Ok(outcome) => outcome,
            Err(err) => {
                if created_root
                    && matches!(self.node(root_id).map(|n| n.is_leaf()), Ok(true))
                {
                    let _ = self.drop_namespace_root(root_id);
                }
                return Err(err);
            }
        };

        self.inserted_pages += inserted_pages;
        let handle = self.make_handle(node_id)?;
        if handle.cached_tokens != (cacheable_pages * self.page_size) as u64 {
            return msg("inserted handle length is inconsistent");
        }
        if pin {
            self.pin_node(node_id)?;
        }
        self.touch_path(node_id);
        if let Some((replace_node_id, _)) = replace {
            self.unpin_node(replace_node_id)?;
        }
        Ok(RadixInsertOutcome {
            handle: Some(handle),
            already_cached_pages: (cacheable_pages - inserted_pages as usize) as u64,
            inserted_pages,
        })
    }

    fn insert_walk(
        &mut self,
        root_id: u64,
        cacheable_pages: usize,
        pages: &[PageKey],
        physical_pages: &[i64],
        replace: Option<(u64, u64)>,
        retain: &mut PageHook,
        release: &mut PageHook,
    ) -> RResult<(u64, u64)> {
        let mut node = root_id;
        let mut page_offset = 0usize;
        let mut inserted_pages: u64 = 0;
        if let Some((replace_node_id, replace_cached_tokens)) = replace {
            node = replace_node_id;
            if self.find_root(node)? != root_id {
                return msg("replacement radix handle belongs to another namespace");
            }
            if self.node(node)?.direct_pins <= 0 {
                return msg("replacement radix handle must remain pinned");
            }
            page_offset = (replace_cached_tokens as usize) / self.page_size;
            if page_offset > cacheable_pages {
                return msg("replacement radix handle exceeds the inserted prefix");
            }
            if page_offset != self.node(node)?.full_page_ids.len() {
                return msg("replacement radix handle path length is inconsistent");
            }
            if !self.trust_page_ownership {
                self.validate_handle_key(node, pages, page_offset)?;
            }
        }
        while page_offset < cacheable_pages {
            let child_id = {
                let node_ref = self.node(node)?;
                node_ref.children.get(&pages[page_offset]).copied()
            };
            let child_id = match child_id {
                Some(child_id) => child_id,
                None => {
                    let suffix_key: Vec<PageKey> = pages[page_offset..].to_vec();
                    let suffix_pages: Vec<i64> = physical_pages[page_offset..].to_vec();
                    if !self.trust_page_ownership {
                        self.validate_new_page_ids(node, &suffix_pages)?;
                    }
                    retain(&suffix_pages).map_err(|()| RadixErr::Callback)?;
                    match self.new_node(node, suffix_key.clone(), suffix_pages.clone()) {
                        Ok(new_id) => {
                            inserted_pages += self.node(new_id)?.page_ids.len() as u64;
                            node = new_id;
                            break;
                        }
                        Err(err) => {
                            // Mirror the Python contract: a failed edge
                            // creation releases the pages it just retained.
                            let _ = release(&suffix_pages);
                            return Err(err);
                        }
                    }
                }
            };
            let common = {
                let child = self.node(child_id)?;
                Self::common_prefix_pages(
                    &child.token_pages,
                    pages,
                    page_offset,
                    cacheable_pages,
                )
            };
            if common == 0 {
                return msg("radix child key does not match its edge");
            }
            if common < self.node(child_id)?.page_ids.len() {
                node = self.split_node(child_id, common)?;
                page_offset += common;
                continue;
            }
            node = child_id;
            page_offset += common;
        }
        Ok((node, inserted_pages))
    }

    pub fn pin(&mut self, node_id: u64) -> RResult<()> {
        let node = self.node(node_id)?;
        let _ = node;
        self.pin_node(node_id)
    }

    pub fn unpin(&mut self, node_id: u64) -> RResult<()> {
        self.unpin_node(node_id)
    }

    pub fn evict_pages(
        &mut self,
        target_pages: usize,
        release: &mut PageHook,
    ) -> RResult<Vec<i64>> {
        let mut evicted: Vec<i64> = Vec::new();
        let mut pinned_entries: Vec<(u64, u64, u64)> = Vec::new();
        while evicted.len() < target_pages {
            let node_id = match self.pop_evictable_leaf(&mut pinned_entries) {
                Some(node_id) => node_id,
                None => break,
            };
            let parent = self.node(node_id)?.parent.expect("leaf has a parent");
            let page_ids = self.node(node_id)?.page_ids.clone();
            if let Err(()) = release(&page_ids) {
                if let Some(node) = self.nodes.get(&node_id) {
                    self.lru_heap.push(Reverse((
                        node.last_access,
                        node.lru_version,
                        node.node_id,
                    )));
                }
                return Err(RadixErr::Callback);
            }
            let edge_key = self.node(node_id)?.token_pages[0].clone();
            let full_len = self.node(node_id)?.full_page_ids.len() as i64;
            let page_count = page_ids.len() as i64;
            self.node_mut(parent)?
                .children
                .remove(&edge_key)
                .ok_or_else(|| RadixErr::Msg("radix child is not registered".into()))?;
            self.nodes.remove(&node_id);
            evicted.extend_from_slice(&page_ids);
            self.cached_pages -= page_count;
            self.flattened_page_refs -= full_len;
            self.evictions += 1;
            self.evicted_pages += page_count as u64;
            let parent_is_root_leaf = {
                let parent_node = self.node(parent)?;
                parent_node.is_root() && parent_node.is_leaf()
            };
            if parent_is_root_leaf {
                self.drop_namespace_root(parent)?;
            } else if self.node(parent)?.is_leaf() {
                self.touch(parent);
            }
        }
        for entry in pinned_entries {
            self.lru_heap.push(Reverse(entry));
        }
        self.maybe_compact_lru();
        Ok(evicted)
    }

    pub fn reset(&mut self, release: &mut PageHook) -> RResult<()> {
        if self.nodes.values().any(|node| node.direct_pins != 0) {
            return msg("cannot reset radix cache with pinned handles");
        }
        let pages = self.cached_page_ids();
        if !pages.is_empty() {
            release(&pages).map_err(|()| RadixErr::Callback)?;
        }
        if let Some(root) = self.nodes.get_mut(&self.root) {
            root.children.clear();
        }
        self.roots = HashMap::from([("".to_string(), self.root)]);
        self.root_namespaces = HashMap::from([(self.root, "".to_string())]);
        self.nodes = HashMap::from([(self.root, RadixNode::new_root(self.root))]);
        self.lru_heap.clear();
        self.cached_pages = 0;
        self.flattened_page_refs = 0;
        self.pinned_pages = 0;
        Ok(())
    }

    pub fn assert_integrity(&self) -> RResult<()> {
        if self.roots.get("").copied() != Some(self.root) {
            return msg("default radix root is invalid");
        }
        if self.root_namespaces.len() != self.roots.len()
            || self
                .roots
                .iter()
                .any(|(namespace, id)| self.root_namespaces.get(id) != Some(namespace))
        {
            return msg("radix namespace directory is inconsistent");
        }
        if self.roots.values().collect::<std::collections::HashSet<_>>().len()
            != self.roots.len()
        {
            return msg("radix namespaces share a root");
        }
        for root_id in self.roots.values() {
            let root = self.node(*root_id)?;
            if root.parent.is_some() || !root.page_ids.is_empty() {
                return msg("radix root metadata is invalid");
            }
        }
        let mut visited: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut page_ids: std::collections::HashSet<i64> = std::collections::HashSet::new();
        let mut subtree_result: HashMap<u64, i64> = HashMap::new();
        for root_id in self.roots.values() {
            let mut stack: Vec<(u64, bool)> = vec![(*root_id, false)];
            while let Some((id, processed)) = stack.pop() {
                let node = self.node(id)?;
                if processed {
                    let mut subtree = node.direct_pins;
                    for child_id in node.children.values() {
                        subtree += subtree_result.get(child_id).copied().unwrap_or(0);
                    }
                    if node.is_root() {
                        if node.subtree_pins != 0 {
                            return msg("radix root must not be pinned");
                        }
                    } else if node.subtree_pins != subtree {
                        return msg("radix subtree pin count is inconsistent");
                    }
                    subtree_result.insert(id, subtree);
                    continue;
                }
                if !visited.insert(id) {
                    return msg("radix tree contains a cycle");
                }
                if node.token_pages.len() != node.page_ids.len() {
                    return msg("radix edge token/page lengths differ");
                }
                if node.token_pages.iter().any(|page| page.len() != self.page_size) {
                    return msg("radix edge contains a partial token page");
                }
                if node.page_ids.iter().any(|page| page_ids.contains(page)) {
                    return msg("physical page appears in multiple radix edges");
                }
                for page in &node.page_ids {
                    page_ids.insert(*page);
                }
                let expected_full: Option<Vec<i64>> = if node.is_root() {
                    Some(node.page_ids.clone())
                } else {
                    node.parent
                        .and_then(|parent| self.nodes.get(&parent))
                        .map(|parent| {
                            let mut full = parent.full_page_ids.clone();
                            full.extend_from_slice(&node.page_ids);
                            full
                        })
                };
                if expected_full.as_ref() != Some(&node.full_page_ids) {
                    return msg("radix flattened page path is inconsistent");
                }
                stack.push((id, true));
                for (key, child_id) in node.children.iter() {
                    let child = match self.nodes.get(child_id) {
                        Some(child) => child,
                        None => return msg("radix child is not registered"),
                    };
                    if child.parent != Some(id) {
                        return msg("radix parent pointer is invalid");
                    }
                    if child.token_pages.is_empty() || child.token_pages[0] != *key {
                        return msg("radix child key is invalid");
                    }
                    stack.push((*child_id, false));
                }
            }
        }
        if visited.len() != self.nodes.len() {
            return msg("radix node directory contains detached nodes");
        }
        if page_ids.len() as i64 != self.cached_pages {
            return msg("radix cached-page accounting is inconsistent");
        }
        let flattened_page_refs: i64 = self
            .nodes
            .values()
            .map(|node| node.full_page_ids.len() as i64)
            .sum();
        if flattened_page_refs != self.flattened_page_refs {
            return msg("radix flattened-page accounting is inconsistent");
        }
        let pinned_pages: i64 = self
            .nodes
            .values()
            .filter(|node| node.subtree_pins > 0)
            .map(|node| node.page_ids.len() as i64)
            .sum();
        if pinned_pages != self.pinned_pages {
            return msg("radix pinned-page accounting is inconsistent");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache() -> RadixPrefixCache {
        RadixPrefixCache::new(2, true).expect("valid page size")
    }

    /// Split flat token ids into full pages of `page_size` tokens.
    fn pages(tokens: &[i64]) -> Vec<PageKey> {
        tokens
            .chunks(2)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    const NO_HOOK: fn(&[i64]) -> Result<(), ()> = |_| Ok(());

    #[test]
    fn insert_match_pin_evict_round_trip() {
        let mut cache = cache();
        cache
            .insert_prefix(4, &pages(&[1, 2, 3, 4]), "", &[10, 11], false, None, &mut NO_HOOK, &mut NO_HOOK)
            .expect("first insert");
        cache
            .insert_prefix(4, &pages(&[1, 2, 3, 4]), "", &[10, 11], false, None, &mut NO_HOOK, &mut NO_HOOK)
            .expect("duplicate insert");

        let outcome = cache
            .insert_prefix(4, &pages(&[5, 6, 7, 8]), "", &[20, 21], false, None, &mut NO_HOOK, &mut NO_HOOK)
            .expect("second insert");
        assert_eq!(outcome.already_cached_pages, 0);
        assert_eq!(outcome.inserted_pages, 2);

        let hit = cache
            .match_prefix(4, &pages(&[1, 2, 3, 4]), "", 0, true)
            .expect("match");
        let handle = hit.handle.expect("full match");
        assert_eq!(handle.cached_tokens, 4);
        assert_eq!(handle.page_ids, vec![10, 11]);
        assert_eq!(cache.cached_pages(), 4);
        assert_eq!(cache.pinned_pages(), 2);

        // Still pinned: the pinned leaf is protected, but the cold sibling
        // leaf is fair game for eviction.
        let evicted = cache.evict_pages(2, &mut NO_HOOK).expect("evict");
        assert_eq!(evicted, vec![20, 21]);

        cache.unpin(handle.node_id).expect("unpin");
        // Unpinned, node 1 is now evictable; only 2 pages remain cached.
        let evicted = cache.evict_pages(4, &mut NO_HOOK).expect("evict");
        assert_eq!(evicted, vec![10, 11]);
        assert_eq!(cache.cached_pages(), 0);
        cache.assert_integrity().expect("integrity after evict");
    }

    #[test]
    fn partial_overlap_splits_edges() {
        let mut cache = cache();
        cache
            .insert_prefix(4, &pages(&[1, 2, 3, 4]), "", &[10, 11], false, None, &mut NO_HOOK, &mut NO_HOOK)
            .expect("first insert");
        cache
            .insert_prefix(4, &pages(&[1, 2, 5, 6]), "", &[12, 13], false, None, &mut NO_HOOK, &mut NO_HOOK)
            .expect("second insert");
        cache.assert_integrity().expect("integrity after split");

        let hit = cache
            .match_prefix(4, &pages(&[1, 2, 5, 6]), "", 0, false)
            .expect("match second prefix");
        // The shared (1,2) page comes from the first prefix; only (5,6) is
        // owned by this insert's physical page list.
        assert_eq!(hit.handle.expect("hit").page_ids, vec![10, 13]);

        let hit = cache
            .match_prefix(4, &pages(&[1, 2, 3, 4]), "", 0, false)
            .expect("match first prefix");
        assert_eq!(hit.handle.expect("hit").page_ids, vec![10, 11]);
    }

    #[test]
    fn namespaces_are_isolated() {
        let mut cache = cache();
        cache
            .insert_prefix(4, &pages(&[1, 2, 3, 4]), "ns-a", &[10, 11], false, None, &mut NO_HOOK, &mut NO_HOOK)
            .expect("insert ns-a");
        cache
            .insert_prefix(4, &pages(&[1, 2, 3, 4]), "ns-b", &[20, 21], false, None, &mut NO_HOOK, &mut NO_HOOK)
            .expect("insert ns-b");

        let hit = cache
            .match_prefix(4, &pages(&[1, 2, 3, 4]), "ns-a", 0, false)
            .expect("match ns-a");
        assert_eq!(hit.handle.expect("hit").page_ids, vec![10, 11]);

        let hit = cache
            .match_prefix(4, &pages(&[1, 2, 3, 4]), "ns-b", 0, false)
            .expect("match ns-b");
        assert_eq!(hit.handle.expect("hit").page_ids, vec![20, 21]);

        let miss = cache
            .match_prefix(4, &pages(&[1, 2, 3, 4]), "ns-c", 0, false)
            .expect("unknown namespace is a miss");
        assert!(miss.handle.is_none());
        cache.assert_integrity().expect("integrity");
    }

    #[test]
    fn reserve_tokens_limits_the_match() {
        let mut cache = cache();
        cache
            .insert_prefix(4, &pages(&[1, 2, 3, 4]), "", &[10, 11], false, None, &mut NO_HOOK, &mut NO_HOOK)
            .expect("insert");
        // One reserved token truncates the eligible prefix to one page.
        let hit = cache
            .match_prefix(4, &pages(&[1, 2, 3, 4]), "", 1, false)
            .expect("match");
        assert_eq!(hit.eligible_tokens, 2);
        assert_eq!(hit.handle.expect("hit").page_ids, vec![10]);
        // Two reserved tokens leave nothing matchable.
        let miss = cache
            .match_prefix(4, &pages(&[1, 2, 3, 4]), "", 3, false)
            .expect("match");
        assert!(miss.handle.is_none());
    }

    #[test]
    fn evicts_coldest_leaf_first() {
        let mut cache = cache();
        cache
            .insert_prefix(2, &pages(&[1, 2]), "", &[10], false, None, &mut NO_HOOK, &mut NO_HOOK)
            .expect("insert old");
        cache
            .insert_prefix(2, &pages(&[3, 4]), "", &[11], false, None, &mut NO_HOOK, &mut NO_HOOK)
            .expect("insert new");
        let evicted = cache.evict_pages(1, &mut NO_HOOK).expect("evict");
        assert_eq!(evicted, vec![10]);
        assert_eq!(cache.cached_pages(), 1);
        cache.assert_integrity().expect("integrity");
    }
}
