# SignalScope

**Telling Real From Synthetic in the Age of Generative Media**

SignalScope is a computer-vision system for assessing whether an input image is **likely authentic / real** or **likely AI-generated**. It is designed around the generalisation problem: performance should not depend only on generators seen during training.

For each image, SignalScope returns:

- a calibrated `P(AI-generated)` likelihood;
- a threshold-based verdict;
- confidence for the predicted class; and
- an influence-based model-evidence overlay showing regions whose masking most changed the model score.

SignalScope is an analytical aid, not provenance certification. Its output is deliberately phrased as a likelihood assessment rather than proof of image origin.

## Submission Materials

- 📄 **One-Page Model Report:** [View Report](https://docs.google.com/document/d/1ZxnaEloS0hog-_vpLrDbG8kZhfRy9NFg/edit?usp=sharing&ouid=109365953735455283867&rtpof=true&sd=true)
- 🎥 **Demo video:** [View Video](https://drive.google.com/file/d/1CQ8vyEp9nlhbAnBOY2MO4nrUXoDOK8mk/view?usp=sharing)
- 📊 **Project Presentation:** [View Presentation](https://docs.google.com/presentation/d/1jEHCZn531zbrn74UwRwZmVnqd0_XOHPk/edit?usp=sharing&ouid=109365953735455283867&rtpof=true&sd=true)

## What we built

### Core task

- Binary real-vs-AI-generated image classification.
- Single-image prediction interface through a local web application.
- Validation-only calibration and threshold selection.
- Generalisation-first evaluation with a generator family held out from training/calibration as an internal pseudo-unseen test.

### Implemented extensions

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
