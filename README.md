# SignalScope

Telling Real From Synthetic in the Age of Generative Media.

SIH 2026 — Internal Hackathon

## Status

Project setup in progress.

## Core Task

Classify an input image as:

- Real
- AI-generated

The system will ultimately report a calibrated confidence score and be evaluated on the organizer-provided held-out test set, including images from unseen generators.

## Planned Architecture

Image
→ Preprocessing
→ Stream A: Frozen CLIP + classifier
→ Stream B: Frequency-domain CNN
→ Fusion + Calibration
→ Verdict + Confidence
→ Explanation
→ User Interface

## Planned Bonus Modules

- Faithful Explanation
- Robustness to Degradation
- Real-Time / Deployable Interface

## Team

SIH 2026 SignalScope Team
