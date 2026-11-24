# Security Policy

## Supported versions

| Version | Supported |
| --- | --- |
| 0.2.x | ✅ |
| < 0.2 | ❌ |

Security fixes are made on the latest release line only. Please upgrade before
reporting an issue against an older version.

## Reporting a vulnerability

Please **do not** open a public issue for security vulnerabilities. Use GitHub's
private vulnerability reporting for this repository instead:

**https://github.com/ohtensorplay/megalm/security/advisories/new**

Include as much of the following as you can:

- Affected version / commit and the component involved (gateway, kernel,
  weight downloader, protocol parser, ...)
- Steps to reproduce and a minimal proof of concept
- Impact (e.g. remote code execution, path traversal, information disclosure,
  denial of service) and whether it is remotely reachable
- Any suggested mitigation or fix

You should receive an acknowledgement within 72 hours. We will keep you
informed as we investigate, and will coordinate disclosure and a fix with you
before publishing an advisory.

## Scope notes

- Model weights and tokenizers under `model/` are third-party artifacts and are
  not part of this repository's security scope.
- Report vulnerabilities in TensorPlay itself to the
  [TensorPlay project](https://github.com/lexing-2026/TensorPlay/security/advisories/new).
- General bugs that are not security-sensitive belong in the
  [issue tracker](https://github.com/ohtensorplay/megalm/issues/new/choose).
