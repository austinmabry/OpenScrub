# License-plate detection (optional)

OpenScrub can blur license plates in video, but — unlike face detection — the
plate model is **not bundled**. Plate detection is opt-in: without a model, the
`plate` category is simply inactive (it never errors, it just finds nothing).

## Why no bundled model

A good plate detector is a trained neural network (tens of MB) with its own
license terms, and plate appearance varies enough by region that no single small
model is universally "right." Rather than ship one and imply a guarantee,
OpenScrub lets you drop in the model that fits your footage and licensing needs.

## What OpenScrub expects

A **single-class YOLOv8 detector exported to ONNX**:

- input shape `(1, 3, 640, 640)`
- output shape `(1, 5, 8400)`  (one class: license-plate)

This is the default export from Ultralytics YOLOv8
(`model.export(format="onnx", imgsz=640, opset=12)`). OpenScrub runs it through
OpenCV's DNN module, so **no PyTorch or ultralytics runtime dependency** is
needed to *use* it — only to train/export one.

## Getting a model

Any of these produce a compatible file:

1. **Train/fine-tune your own** on a plate dataset (e.g. the Roboflow
   "License Plate Recognition" datasets) with Ultralytics YOLOv8, then export to
   ONNX as above. This gives you a model tuned to your region and license terms.
2. **Convert an existing YOLOv8 `.pt` plate model** to ONNX with the one-line
   export above.
3. **Use your own** ONNX plate detector if it matches the input/output shapes.

## Installing the model

Put the file at either location (OpenScrub checks both):

    models/plate_yolov8.onnx        (next to openscrub.py)
    plate_yolov8.onnx

…or point at it explicitly:

    export OPENSCRUB_PLATE_MODEL=/path/to/your_plate_model.onnx
    # or per-run:
    openscrub input.mp4 --categories plate --plate-model /path/to/model.onnx

`install.py --with-plates` can fetch a model for you if you supply a source with
`--plate-model-url` (no default URL ships, so you choose the model and accept its
license).

## Using it

    # CLI
    openscrub dashcam.mp4 --categories plate,face --plate-threshold 0.35

    # Web UI: check the "plate" category. If no model is loaded, the run log
    # will say "plate detector: no model found" and plates won't be touched.

Plates are treated like any other category: **zoneable** (draw a plate zone to
restrict where it looks), and they honor **per-category redaction mode**
(blur / box / mosaic). Because plates on dashcam/CCTV footage move fast,
OpenScrub re-detects them **every frame** (like dense faces), so a plate crossing
the frame stays covered.

## Honest limitations

- Detection quality is entirely the model's. A model trained on EU plates will
  underperform on US plates, angled/distant plates are harder, and motion blur
  hurts. **Always keep the human-review step** for anything sensitive.
- OpenScrub blurs the plate; it does **not** read it. That's deliberate — for
  redaction you only need to cover it, and not reading avoids a whole class of
  OCR-accuracy problems.
