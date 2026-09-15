# SignalScope model report

Complete this report only after a real validation run; do not substitute training accuracy.

| Field | Result |
| --- | --- |
| Task | Binary real-vs-AI-generated image assessment, with a calibrated confidence and a conservative sensitivity overlay. |
| Data and split | CIFAKE train/test are supported. Record exact counts, licenses, and any second source used here. Keep organizer held-out data untouched. |
| Approach | DS-GID: ResNet semantic stream + log-FFT frequency stream + learned interaction fusion + validation-only temperature scaling. |
| Validation metrics | Run `evaluate.py` on exported predictions and paste overall ROC-AUC, macro-F1, accuracy, FPR, threshold, and 95% AUC CI. |
| Unseen-generator result | **Not yet measured.** Use a completely held-out generator family and report its AUC separately. |
| Limitations | A classifier is not provenance proof. It may fail on new generators, edits, compression, camera processing, or distribution shift. |

Method context: Ojha et al., *Towards Universal Fake Image Detectors that Generalize Across Generative Models* (CVPR 2023); Frank et al., *Leveraging Frequency Analysis for Deep Fake Image Recognition* (ICML 2020 workshop).
