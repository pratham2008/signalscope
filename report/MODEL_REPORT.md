# SignalScope Model Report

## 1. Task

SignalScope performs binary image assessment:

- `Likely Authentic / Real`
- `Likely AI-Generated`

The production interface returns a calibrated AI-generation likelihood, a threshold-based verdict, confidence for the predicted class, and an influence-based evidence overlay.

SignalScope is an analytical aid, not provenance certification.

## 2. Data and Split

Current development-scale generalisation manifest:

| Split | Images |
| --- | ---: |
| Training | 22,000 |
| Validation | 16,000 |
| Internal pseudo-unseen VQDM | 1,000 |

Training and validation combine CIFAKE-style data with a generator-diverse GenImage subset.

Seen generator families include BigGAN, Stable Diffusion v5, Wukong, ADM, GLIDE, and Midjourney.

The VQDM generator family is kept outside model training and calibration and is used only as an internal transfer/generalisation proxy.

The organizer's private held-out test set is never used for training, calibration, or threshold selection.

## 3. Production Model

SignalScope uses two complementary streams:

1. Frozen OpenAI CLIP ViT-B/32 with normalized 512-D embeddings.
2. A lightweight trainable CNN over log-scaled 2-D FFT magnitude.

The projected semantic and frequency representations are fused together with an elementwise interaction feature.

Production fusion:

`128 semantic + 128 frequency + 1 interaction -> 257 -> 64 -> 1`

Training configuration:

- Loss: `BCEWithLogitsLoss`
- Optimizer: `AdamW`
- Learning rate: `1e-3`
- Weight decay: `1e-4`
- Epochs: `5`
- Seed: `42`
- Selected augmentation: random-resolution degradation
- Calibration: temperature scaling on validation only
- Operating threshold: `0.646906`

## 4. Validation Results

| Metric | Previous Dual-Stream Baseline | Current Production Model |
| --- | ---: | ---: |
| ROC-AUC | 0.8987 | **0.9669** |
| Accuracy | 77.74% | **88.83%** |
| Macro-F1 | 0.7705 | **0.8879** |
| Validation FPR Target | 5.0% | **5.0%** |

Production-model improvement over the baseline:

- ROC-AUC: `+0.0682` absolute
- Accuracy: `+11.09` percentage points
- Macro-F1: `+0.1173`

Validation confusion matrix:

```text
                 Predicted Real   Predicted AI
Actual Real          7600              400
Actual AI            1387             6613
```

## 5. Internal Pseudo-Unseen Evaluation

VQDM was held outside training and calibration and is used here as an internal generator-shift proxy.

| Metric | VQDM |
| --- | ---: |
| Images | 1,000 |
| ROC-AUC | **0.7276** |
| Accuracy | **57.4%** |
| Macro-F1 | **~0.5106** |
| FPR | **6.6%** |
| Threshold | **0.646906** |

Confusion matrix:

```text
                 Predicted Real   Predicted AI
Actual Real           467               33
Actual AI             393              107
```

This result is not the organizer's official held-out score.

## 6. Explanation Method

The production explanation uses 7x7 occlusion attribution.

Image regions are masked with a blurred replacement and the resulting change in `P(AI-generated)` is measured.

- Positive influence: masking the region lowers `P(AI-generated)`, so the region supports the model's AI-generation score.
- Negative influence: masking the region raises `P(AI-generated)`, so the region opposes the model's AI-generation score.

The visualization represents model sensitivity. It is not causal proof that a highlighted region contains a synthetic artifact.

## 7. Limitations

- Generator shift remains a significant challenge, as shown by the lower internal VQDM result.
- The internal pseudo-unseen evaluation is not an official organizer score.
- Small-resolution stress tests degraded substantially; broad resolution robustness is therefore not claimed.
- AI-image detection should be treated as a likelihood assessment rather than definitive provenance proof.

## 8. Reproducibility

The production checkpoint is included in:

`model/dual/clip_fft_resolution.pt`

The prediction interface is:

```python
from predict import predict

label, confidence = predict("path/to/image.jpg")
```

The organizer-held-out test set remains outside training, calibration, and threshold selection.

## 9. References

1. Bird, J. J. & Lotfi, A. (2023). *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images.* IEEE Access.
2. Zhu, M. et al. (2023). *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image.* NeurIPS.
3. Ojha, U., Li, Y. & Lee, Y. J. (2023). *Towards Universal Fake Image Detectors that Generalize Across Generative Models.* CVPR.
4. Frank, J., Eisenhofer, T., Schönherr, L., Fischer, A., Kolossa, D. & Holz, T. (2020). *Leveraging Frequency Analysis for Deep Fake Image Recognition.* CVPR.
