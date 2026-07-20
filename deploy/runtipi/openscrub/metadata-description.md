# OpenScrub

Local, GPU-capable video redaction. Load a video in the web UI, pick what to
detect, and OpenScrub finds and blurs:

- **Faces** — grouped by identity, so review shows one decision per person
- **Full bodies** — silhouette-precise masking, not big rectangles
- **License plates**
- **On-screen text** — names, SSNs, DOBs, phones, emails, addresses, credit
  cards, API keys, IPs, plus your own custom regex categories
- **Anything else** — draw a box around a person or object and it is tracked
  and blurred through the clip

Every detection is shown for **human review** before rendering — over-blur
beats under-blur, and nothing is trusted without your sign-off. Audio spans
can be muted or bleeped. Job data can be encrypted at rest (AES-256).
Nothing ever leaves your machine.

**Performance note:** scanning is CPU-intensive — fast on x86 and Apple
Silicon, slow (but working) on Raspberry Pi. On x86 with an NVIDIA GPU, use
the `pharmhero/openscrub:cuda` image instead for accelerated OCR/detection.
