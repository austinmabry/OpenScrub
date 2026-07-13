# Security Policy

OpenScrub is a privacy tool: people point it at their most sensitive
footage. Security reports are taken seriously and handled with priority.

## Supported versions

Only the **latest release** receives fixes. There is no LTS line — the
in-app updater, pip, and the Docker `latest`/`cuda` tags all make staying
current a one-step operation.

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Use GitHub's private reporting: **Security tab → Report a vulnerability**
on this repository. You'll get a response within a few days. If the
report is valid, a fix ships as a new release and the advisory is
published after users have had a reasonable window to update.

## What counts as a security issue here

Beyond the usual (RCE, path traversal, authentication bypass in the web
UI, vault crypto weaknesses), OpenScrub treats **redaction failures with
a systematic cause** as security-relevant: a class of input where content
the tool claims to redact is reliably left exposed (not ordinary OCR
misses — the docs are explicit that this is best-effort detection with
mandatory human review, but e.g. "blur boxes are misplaced for all videos
with property X" qualifies).

## Out of scope

- Missed detections on individual videos (that is what the review step
  is for)
- Issues requiring an attacker who already controls the machine or the
  LAN the server was deliberately exposed to
- The self-signed certificate warning (documented behavior; install your
  own certificate in App settings)
