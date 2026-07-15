# OpenScrub server image (CPU). Built and pushed to GitHub Container
# Registry and Docker Hub automatically on every release
# (.github/workflows/docker-image.yml):
#
#     docker run -d -p 8384:8384 \
#       -v openscrub_data:/root/.local/share/OpenScrub \
#       ghcr.io/austinmabry/openscrub:latest        # or pharmhero/openscrub
#
# then open https://<host>:8384/ (self-signed cert; add --token via the
# command below for access control):
#     docker run ... ghcr.io/austinmabry/openscrub:latest \
#       openscrub-web --host 0.0.0.0 --token mysecret
#
# Data (jobs, certs, zones, models, vault) lives in the mounted volume —
# the container itself is disposable. To update: pull the new tag and
# recreate the container (the in-app updater is disabled in Docker).
# Notes: CPU OCR/encode only (no CUDA/NVENC). spaCy NER is included.
#
# LAYER ORDER MATTERS: heavy dependencies install BEFORE the app code is
# copied, so a release that only changes code rebuilds (and users only
# re-download) the small app layers at the bottom. Combined with the
# workflow's registry build cache, unchanged dependency layers keep the
# same digest across releases.

FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# ---- heavy dependency layer: rebuilds ONLY when requirements.txt changes
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 # spaCy NER for name detection, baked in with its model
 && python -m spacy download en_core_web_sm

# ---- app layers: tiny, change every release
COPY pyproject.toml README.md LICENSE NOTICE plate_models.json face_models.json ./
COPY openscrub.py openscrub_web.py openscrub_setup.py openscrub_update.py \
     openscrub_vault.py zones_ui.py install.py test_openscrub.py ./
COPY assets/openscrub.ico assets/
RUN pip install --no-cache-dir --no-deps . \
 # pre-fetch the YuNet face model so first run works offline
 && mkdir -p /root/.openscrub/models \
 && python -c "import urllib.request, openscrub; \
urllib.request.urlretrieve(openscrub.YUNET_URL, \
'/root/.openscrub/models/face_detection_yunet_2023mar.onnx'); \
urllib.request.urlretrieve(openscrub.SFACE_URL, \
'/root/.openscrub/models/face_recognition_sface_2021dec.onnx')"

EXPOSE 8384
VOLUME ["/root/.local/share/OpenScrub"]
CMD ["openscrub-web", "--host", "0.0.0.0"]
