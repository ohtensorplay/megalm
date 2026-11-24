<!-- Thank you for contributing! Please see CONTRIBUTING.md before opening a PR. -->

## Motivation

<!-- What problem does this PR solve, and why is the change needed? -->

## Modifications

<!-- Summarize the changes by component: kernels/, runtime/, gateway/, rust/, docs/, ... -->

## Related Issues

<!-- e.g. "Fixes #123", "Part of #456" -->

## Test Plan

<!-- Which tests did you run? Include commands and results, e.g. `pytest tests -q`, `cargo test`. -->

## Accuracy / Performance Evidence

<!-- For model or kernel changes: accuracy checks. For optimizations: before/after
     numbers, GPU, problem sizes, and the benchmark command. Delete if not applicable. -->

## Checklist

- [ ] The title and description are accurate (they become the squash-commit message).
- [ ] `ruff check src tests` and `python -m pytest tests -q` pass locally.
- [ ] `cargo check --manifest-path rust/Cargo.toml --locked` passes for Rust changes.
- [ ] Tests and documentation were added or updated as needed.
- [ ] A user-visible change is recorded under `Unreleased` in `CHANGELOG.md`.

## Reviewer Notes

<!-- Optional: areas you would like reviewers to focus on, known limitations. -->
