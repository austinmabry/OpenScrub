# OpenScrub Intel

OpenScrub is a local video-redaction tool: it detects and blurs faces,
whole people (silhouette body masking), license plates, and on-screen PII
(names, SSNs, addresses, card numbers, custom regex) in videos and screen
recordings, with a human review step before anything is trusted. Runs
entirely on this server — no footage leaves your network.

Intel GPU build: Quick Sync hardware encoding and OpenVINO-accelerated detection on any Intel iGPU (Gen 9+) or Arc card. Allocate the GPU in the Resources section of the install form; the image ships Intel’s media driver, so the host needs no extra driver install.

The web interface serves self-signed HTTPS on port 8384 by default; the
browser shows a one-time certificate warning. Detection models download
on first use and persist in the app's data storage. Optional read-only
media mounts (Additional Storage) let the in-app "path on the server"
picker process files without uploading them through the browser.
