# Third-party notices

OpenScrub's own code is licensed under the Apache License 2.0 — see
[LICENSE](LICENSE) and [NOTICE](NOTICE).

This file covers the third-party **model files** OpenScrub uses. Each is a
separate work under its own license. OpenScrub runs alongside these files;
it does not relicense them, and they do not change OpenScrub's license.

**How these files reach you**

- **Docker images:** every file in the table below is pre-installed in
  `/root/.openscrub/models` (the CUDA image omits the two PP-OCRv5
  recognition files). Because the images redistribute these files, their
  license texts and copyright notices are reproduced in full below. In
  the images, this file is at `/src/THIRD_PARTY_NOTICES.md`.
- **pip, the Windows installer, and source installs:** no model files are
  bundled. OpenScrub downloads each file on first use directly from the
  upstream source URL listed below and verifies its SHA-256 before use.

Licenses and copyright lines were taken from the upstream license files
and model pages, checked on 2026-11-05.

## Summary

| File (as installed) | Component | License | Copyright / attribution |
|---|---|---|---|
| `face_detection_yunet_2023mar.onnx` | YuNet face detector | MIT | Copyright (c) 2020 Shiqi Yu |
| `face_recognition_sface_2021dec.onnx` | SFace face recognition | Apache-2.0 | contributed by Yaoyao Zhong; ONNX conversion by Chengrui Wang |
| `object_tracking_vittrack_2023sep.onnx` | VitTrack object tracker | Apache-2.0 | contributed by Pengyu Liu (GSoC 2023) |
| `text_detection_ppocrv5_mobile.onnx` | PP-OCRv5 mobile text detection | Apache-2.0 | PaddlePaddle Authors |
| `text_recognition_ppocrv5_mobile.onnx` | PP-OCRv5 mobile text recognition | Apache-2.0 | PaddlePaddle Authors |
| `text_recognition_ppocrv5_mobile.yml` | PP-OCRv5 recognition config + character dictionary | Apache-2.0 | PaddlePaddle Authors |
| `oim-yolov9-t-640.onnx` | YOLOv9-T 640 license-plate detector | MIT | Copyright (c) 2024 ankandrew |

## Details

### YuNet face detector

- **File:** `face_detection_yunet_2023mar.onnx`
  (SHA-256 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`)
- **Source:** <https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx>
- **License:** MIT — full text below, from
  <https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE>
- **Copyright:** Copyright (c) 2020 Shiqi Yu <shiqi.yu@gmail.com>
- **Credit:** model from <https://github.com/ShiqiYu/libfacedetection.train>.
  Paper: W. Wu, H. Peng, S. Yu, "YuNet: A Tiny Millisecond-level Face
  Detector", Machine Intelligence Research, 2023.

### SFace face recognition

- **File:** `face_recognition_sface_2021dec.onnx`
  (SHA-256 `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`)
- **Source:** <https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx>
- **License:** Apache License 2.0 — full text below, from
  <https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/LICENSE>.
  The upstream README states that all files in that directory are
  licensed under Apache 2.0.
- **Copyright:** the upstream license file does not name a copyright
  holder, and the upstream project ships no NOTICE file.
- **Credit:** SFace is contributed by Yaoyao Zhong
  (original code: <https://github.com/zhongyy/SFace>); ONNX conversion by
  Chengrui Wang. Paper: "SFace: Sigmoid-Constrained Hypersphere Loss for
  Robust Face Recognition" (<https://arxiv.org/abs/2205.12010>).

### VitTrack object tracker

- **File:** `object_tracking_vittrack_2023sep.onnx`
  (SHA-256 `2990f0b7cd44d92afa48cd97db6de7be113fc1d9594fddb74e2725c10478e91d`)
- **Source:** <https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx>
- **License:** Apache License 2.0 — full text below, from
  <https://github.com/opencv/opencv_zoo/blob/main/models/object_tracking_vittrack/LICENSE>.
  The upstream README states that all files in that directory are
  licensed under Apache 2.0.
- **Copyright:** the upstream license file does not name a copyright
  holder, and the upstream project ships no NOTICE file.
- **Credit:** contributed by Pengyu Liu in the Google Summer of Code 2023
  project "Realtime object tracking models"; based on OSTrack
  (<https://github.com/botaoye/OSTrack>).

### PP-OCRv5 mobile text detection and recognition

- **Files:**
  - `text_detection_ppocrv5_mobile.onnx`
    (SHA-256 `a431985659dc921974177a95adcfbb90fd9e51989a5e04d70d0b75f597b6e61d`),
    from <https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det_onnx/resolve/main/inference.onnx>
  - `text_recognition_ppocrv5_mobile.onnx`
    (SHA-256 `da72dc72ca4dc220df0dfde68c1dedc31c58d3e76a25871122e5056227d50092`),
    from <https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec_onnx/resolve/main/inference.onnx>
  - `text_recognition_ppocrv5_mobile.yml`
    (SHA-256 `5dfeb2777f6d0db8177d8128a8acfcf6e6276dc4ac73ea3bf0dc06d6a5e85d8e`),
    from <https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec_onnx/resolve/main/inference.yml>
- **License:** Apache License 2.0 — full text below. Both Hugging Face
  model cards (<https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det_onnx>,
  <https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec_onnx>) declare
  `license: apache-2.0`; the model repositories contain no separate
  license file.
- **Copyright:** Copyright (c) 2016 PaddlePaddle Authors. All Rights
  Reserved. — as stated in the PaddleOCR project's license
  (<https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE>). The
  upstream projects ship no NOTICE file.
- **Credit:** PaddleOCR, <https://github.com/PaddlePaddle/PaddleOCR>.

### YOLOv9-T 640 license-plate detector

- **File:** `oim-yolov9-t-640.onnx`
  (SHA-256 `c3c1026ca7d0585dd88084d68182dd897113712fa734ae1557ca70174440c076`)
- **Source:** <https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-t-640-license-plates-end2end.onnx>
- **License:** MIT — full text below, from
  <https://github.com/ankandrew/open-image-models/blob/main/LICENSE>
- **Copyright:** Copyright (c) 2024 ankandrew
- **Credit:** open-image-models, <https://github.com/ankandrew/open-image-models>.
- **Training data:** the upstream project does not publish its training
  dataset or that dataset's license (its README, documentation site, and
  model release notes were checked).

## Models downloaded only on request (not redistributed)

These are never bundled in any OpenScrub package or image. They are
downloaded from their upstream source only when you choose them in
Settings → Detection models (or use the feature that needs them), and they
reach you directly from that source under its own license. Each model's
license is shown beside it in the Detection models panel.

- **Other license-plate models** (`plate_models.json`): open-image-models,
  MIT, Copyright (c) 2024 ankandrew.
- **Optional face models** (`face_models.json`): CenterFace (MIT) and
  SCRFD (non-commercial license — never selected automatically).
- **Person / silhouette models** (`person_models.json`): YOLO11 and
  YOLOv8 segmentation models, AGPL-3.0 — download only, never bundled and
  never downloaded automatically.
- **Speech-recognition models** for the spoken-PII feature, fetched by
  the faster-whisper package on first use under the license shown on
  each model's page.

## Other third-party software in the Docker images

The Docker images also contain third-party software (FFmpeg, Tesseract
OCR, OpenCV, PaddleOCR, spaCy and its English model, onnxruntime,
faster-whisper, and others). These are installed with the operating
system's package manager or with pip, which install each package's own
license files alongside it (under `/usr/share/doc/<package>/` for Debian
packages, and in each Python package's `.dist-info` directory). They are
aggregated alongside OpenScrub, not part of it.

## License texts

### MIT License — YuNet

```
MIT License

Copyright (c) 2020 Shiqi Yu <shiqi.yu@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### MIT License — open-image-models

```
MIT License

Copyright (c) 2024 ankandrew

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Apache License 2.0 — SFace, VitTrack, PP-OCRv5 models

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
