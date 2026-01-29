# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly:

1. **Do not** open a public GitHub issue
2. Email the maintainer or use [GitHub's private vulnerability reporting](https://github.com/Osman-Geomatics93/crop-classification-deep-learning/security/advisories/new)
3. Include a description of the vulnerability and steps to reproduce

You can expect an initial response within 48 hours.

## Scope

This project processes satellite imagery and training data locally. It does not include:
- Web services or APIs
- User authentication
- Network-facing components

Security concerns are primarily related to:
- Dependency vulnerabilities (monitored by Dependabot)
- Safe handling of file paths in data processing scripts
