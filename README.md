# SignalScope

SignalScope estimates whether an image is likely authentic or AI-generated. It is designed around the difficult part of the task: generalizing beyond a single generator family. It returns a calibrated probability, an operating-point verdict, and an optional model-sensitivity overlay. It does **not** establish provenance or authenticity with certainty.

## What is implemented

`DS-GID` uses two complementary signals:

1. A semantic ResNet-18 stream, optionally initialized from frozen ImageNet weights.
2. A log-scaled 2D FFT stream that learns frequency-domain signals.
3. A learned fusion head over both logits and their interaction.
4. Temperature scaling and a validation-selected threshold at a fixed real-image false-positive-rate target.

The repository also includes degradation augmentation, a single-image CLI, a Gradio demo, saliency overlays, prediction export, and a reproducible metric/reporting harness. The original `CoreDetector` checkpoint remains supported as a legacy baseline.

## Quick start

Python 3.10+ is recommended. From the project root:

```bash
bash setup.sh
source .venv/bin/activate
```

The current layout already expects CIFAKE-style folders:

```text
data/raw/cifake/
  train/{REAL,FAKE}/
  test/{REAL,FAKE}/
```

For a CPU smoke run (not a quality run):

```bash
python -m src.train_dual --epochs 1 --train-limit 100 --val-limit 50 --batch-size 16
python predict.py data/raw/cifake/test/REAL/<an-image>.jpg
```

For an actual training run, use the entire training data and select the validation directory as a generator family or source that is excluded from training whenever possible:

```bash
python -m src.train_dual \
  --train-dir data/raw/cifake/train \
  --val-dir data/raw/cifake/test \
  --epochs 8 --batch-size 32 --target-fpr 0.05
```

`--pretrained-semantic --freeze-semantic` uses ResNet ImageNet weights; torchvision downloads them the first time. Default training is offline and trains the semantic stream from scratch. For a serious unseen-generator result, train on at least two source families and reserve a whole generator family for validation; do not tune on the organizer’s hidden test set.

## Predict and explain

```bash
python predict.py path/to/image.jpg \
  --model model/dual/signalscope_dual_stream.pt \
  --explanation-output report/examples/image_overlay.png
```

The public API is:

```python
from predict import predict
label, confidence = predict("path/to/image.jpg")
```

The `--explanation-output` image shows sensitivity, not a confirmed artifact location. The text deliberately uses hedged language.

To run the local demo after installing optional UI dependencies:

```bash
python app.py
```

## Evaluate a checkpoint

Export labeled predictions, then evaluate that immutable CSV. This cleanly separates inference from reporting.

```bash
python scripts/export_predictions.py \
  --data-dir data/raw/cifake/test \
  --model model/dual/signalscope_dual_stream.pt \
  --output report/predictions.csv --generator cifake

python evaluate.py report/predictions.csv --output-dir report/evaluation --target-fpr 0.05
```

The output contains ROC-AUC, macro-F1, confusion matrix, accuracy, FPR at the stated threshold, bootstrap AUC confidence interval, optional per-generator AUC, and plots when Matplotlib is installed. `report/MODEL_REPORT.md` is intentionally a template: do not claim metrics until this command has produced them.

## Reproducibility and data discipline

- Record every source, license, sample count, and generator split in the completed model report.
- Keep organizer-provided held-out images out of training, calibration, and threshold selection.
- Calibrate temperature and select the FPR threshold on validation only.
- Test recompression, resize, blur, and noise separately before claiming robustness.
- Check false positives manually; a real image incorrectly flagged as synthetic can cause harm.

## Layout

```text
src/train_dual.py                 dual-stream training and checkpoint packaging
src/core/dual_model.py            semantic, FFT, and fusion architecture
src/core/calibration.py           validation-only temperature scaling
src/core/metrics.py               portable metrics with no sklearn dependency
scripts/export_predictions.py     checkpoint-to-CSV inference
evaluate.py                       metrics, confidence interval, optional plots
predict.py                        required single-image prediction interface
app.py                            optional Gradio demonstration
report/MODEL_REPORT.md            submission-ready reporting template
```

## References

- Ojha et al. (2023), *Towards Universal Fake Image Detectors that Generalize Across Generative Models*.
- Frank et al. (2020), *Leveraging Frequency Analysis for Deep Fake Image Recognition*.
- CIFAKE dataset: cite the exact version and license used in your completed report.
