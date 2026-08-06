# OpenScrub Nvidia

OpenScrub is a local video-redaction tool: it detects and blurs faces,
whole people (silhouette body masking), license plates, and on-screen PII
(names, SSNs, addresses, card numbers, custom regex) in videos and screen
recordings, with a human review step before anything is trusted. Runs
entirely on this server — no footage leaves your network.

NVIDIA GPU build: GPU-accelerated OCR (PaddleOCR-CUDA), GPU face detection and identity grouping (CUDA OpenCV), GPU license-plate/person detection (onnxruntime CUDA), and NVENC hardware encoding. Works on Maxwell and newer. Allocate the GPU in the Resources section of the install form.

The web interface serves self-signed HTTPS on port 8384 by default; the
browser shows a one-time certificate warning. Detection models download
on first use and persist in the app's data storage. Optional read-only
media mounts (Additional Storage) let the in-app "path on the server"
picker process files without uploading them through the browser.
