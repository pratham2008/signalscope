# SignalScope

**Telling Real From Synthetic in the Age of Generative Media**

SignalScope is a computer-vision system for assessing whether an input image is **likely authentic / real** or **likely AI-generated**. It is designed around the generalisation problem: performance should not depend only on generators seen during training.

For each image, SignalScope returns:

- a calibrated `P(AI-generated)` likelihood;
- a threshold-based verdict;
- confidence for the predicted class; and
- an influence-based model-evidence overlay showing regions whose masking most changed the model score.

SignalScope is an analytical aid, not provenance certification. Its output is deliberately phrased as a likelihood assessment rather than proof of image origin.

## What we built

**Core task**

- Binary real-vs-AI-generated image classification.
- Single-image prediction interface through a local web application.
- Validation-only calibration and threshold selection.
- Generalisation-first evaluation with a generator family held out from training/calibration as an internal pseudo-unseen test.

**Implemented extensions**

- Influence-based visual explanation for individual predictions.
- Responsive browser UI for upload, assessment, model evidence, and evaluation results.
- JPEG and random-resolution degradation augmentation during selected training.

## Architecture

SignalScope uses two complementary streams:

```text
                         ┌─ Frozen OpenAI CLIP ViT-B/32 ─┐
Input image ─────────────┤   normalized 512-D embedding  ├─> 128-D semantic
                         └──────────────────────────────┘

                         ┌─ Log-scaled 2-D FFT encoder ──┐
Input image ─────────────┤   lightweight trainable CNN   ├─> 128-D frequency
                         └──────────────────────────────┘

                 semantic ⊙ frequency  ──> interaction scalar

             [128 semantic + 128 frequency + 1 interaction]
                              │
                         257 -> 64 -> 1
                              │
                    temperature scaling
                              │
             likelihood + thresholded verdict
                              │
                  influence-based evidence map
```

### Training configuration

- Semantic backbone: frozen OpenAI CLIP ViT-B/32.
- CLIP embedding: normalized 512-D.
- Frequency stream: log-scaled 2-D FFT magnitude.
- Semantic projection: 512 -> 128.
- Frequency projection: 128-D.
- Fusion input: 257-D; fusion head: 257 -> 64 -> 1.
- Loss: `BCEWithLogitsLoss`.
- Optimizer: AdamW.
- Learning rate: `1e-3`.
- Weight decay: `1e-4`.
- Seed: `42`.
- Epochs: `5`.
- Selected run: JPEG augmentation + random-resolution degradation.
- Calibration: temperature scaling on validation only.
- Frozen operating threshold: `0.646906`.

## Data and evaluation protocol

The current development manifest contains:

- **22,000** training images;
- **16,000** validation images; and
- **1,000** internal pseudo-unseen VQDM images.

Training/validation combine CIFAKE-style data with a generator-diverse GenImage subset. The seen generator sources are BigGAN, Stable Diffusion v5, Wukong, ADM, GLIDE, and Midjourney. The VQDM family is kept outside model training and calibration and is used only as an internal transfer/generalisation proxy.

The challenge specification describes a roughly **100k+ labelled training/validation corpus**. The current model is a development-scale run; the next training iteration is intended to scale toward that full labelled training/validation data.

The organizers' private held-out test set is **never used for training, calibration, or threshold selection**. Its official score is not reported here because it is computed by the organizers through the prediction interface during judging.

## Current internal results

### Validation

| Metric | Previous dual-stream baseline | Current champion |
|---|---:|---:|
| ROC-AUC | 0.8987 | **0.9669** |
| Accuracy | 77.74% | **88.83%** |
| Macro-F1 | 0.7705 | **0.8879** |
| Validation FPR target | 5.0% | **5.0%** |

Champion improvement over the baseline:

- `+0.0682` absolute ROC-AUC (`+7.59%` relative);
- `+11.09` percentage points accuracy;
- `+0.1173` Macro-F1.

Validation confusion matrix at the selected operating point:

```text
                 Predicted Real   Predicted AI
Actual Real          7600              400
Actual AI            1387             6613
```

### Internal pseudo-unseen VQDM evaluation

This is an **internal proxy**, not the organizer's official held-out score.

- Images: `1,000` (500 real, 500 synthetic).
- ROC-AUC: **0.7276**.
- Macro-F1: **~0.5106**.
- Accuracy: **57.4%**.
- FPR: **6.6%**.
- Frozen threshold: **0.646906**.

```text
                 Predicted Real   Predicted AI
Actual Real           467               33
Actual AI             393              107
```

The pseudo-unseen result is intentionally reported separately because the challenge's official held-out test remains unavailable to the team until judging.

## Explanation and responsible use

The production explanation is an **influence-based occlusion visualization**. Image regions are masked and the resulting change in `P(AI-generated)` is measured. Regions with positive influence support the model's AI-generation score; regions with negative influence oppose it.

This visualisation localizes **model sensitivity**. It is not causal proof that a highlighted region contains a synthetic artifact.

SignalScope deliberately uses labels such as:

- `Likely AI-Generated`
- `Likely Authentic / Real`

The system should not be used as proof of provenance, identity attribution, or as a definitive judgement about real people, political claims, or real-world events.

### Known limitations

- The internal VQDM pseudo-unseen result is substantially below validation performance, demonstrating a real generator-shift challenge.
- At the frozen threshold on VQDM, only 107 of 500 synthetic images crossed the threshold, while 33 of 500 real images were false positives.
- Separate small-resolution stress tests degraded substantially; broad resolution robustness is therefore **not claimed**.
- No official organizer held-out metric is available before judging.

## Quick start

### 1. Backend

From the project root:

```bash
source .venv/bin/activate
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

The API exposes `/health` and `/predict`.

### 2. Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

Upload a JPEG, PNG, WebP, or BMP image and click **Analyze Image**.

### 3. Direct prediction interface

The repository also exposes the required single-image prediction function:

```python
from predict import predict

label, confidence = predict("path/to/image.jpg")
print(label, confidence)
```

The production checkpoint is:

```text
model/dual/clip_fft_resolution.pt
```

> **Model weights:** upload the checkpoint through a GitHub Release or another stable public download location if it is not committed to the repository, and place that link here: **[MODEL WEIGHTS LINK]**.

## Reproducibility

- Keep organizer-held-out images completely outside training, calibration, and threshold selection.
- Fit temperature scaling and the operating threshold using validation only.
- Record datasets, licences, sample counts, generator identities, and split rules.
- Treat degradation tests separately from clean-set metrics.
- Do not report internal pseudo-unseen results as the official organizer score.

A judge should be able to reproduce a prediction from the README and the released model weights without needing access to the team's development machine.

## Repository layout

```text
.
├── api/
│   └── main.py                         FastAPI prediction service
├── frontend/
│   ├── app/page.tsx                    SignalScope web interface
│   └── app/globals.css                 analytical UI styling
├── src/
│   └── core/
│       ├── dual_model.py               CLIP + FFT fusion model
│       ├── calibration.py              temperature scaling
│       ├── explain.py                  influence-based explanation
│       └── metrics.py                  evaluation metrics
├── scripts/
│   ├── train_clip_fft_resolution.py    champion training pipeline
│   └── evaluate_clip_fft.py            split evaluation
├── model/
│   └── dual/
│       └── clip_fft_resolution.pt       production checkpoint
├── report/
│   └── SignalScope_Model_Report.pdf    one-page model report
├── predict.py                          single-image prediction interface
└── requirements.txt                    Python environment
```

## Demo

**Demo video:** [https://drive.google.com/file/d/1CQ8vyEp9nlhbAnBOY2MO4nrUXoDOK8mk/view?usp=sharing]

The 3–5 minute demonstration shows the core prediction flow, influence-based model evidence, real-vs-AI examples, and the internal pseudo-unseen evaluation snapshot.

## Originality declaration

SignalScope was developed by our team during the SIH 2026 internal hackathon period. We used permitted open-source libraries, a pretrained CLIP backbone, public datasets, and AI coding assistants. No public real-vs-fake notebook was copied wholesale. Third-party datasets, pretrained components, and relevant research are acknowledged below and in the project documentation.

## References

1. Bird, J. J. & Lotfi, A. (2023). **CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images.** IEEE Access.
2. Zhu, M. et al. (2023). **GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image.** NeurIPS.
3. Ojha, U., Li, Y. & Lee, Y. J. (2023). **Towards Universal Fake Image Detectors that Generalize Across Generative Models.** CVPR.
4. Frank, J., Eisenhofer, T., Schönherr, L., Fischer, A., Kolossa, D. & Holz, T. (2020). **Leveraging Frequency Analysis for Deep Fake Image Recognition.** CVPR.
