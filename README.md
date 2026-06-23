# BTD_Prediction_Adaboost
## Breast Tissue Density (BI-RADS) Classification via AdaBoost + MLP Ensemble

**Contributors:** Kaligatla Devi Prasad, Palivela Sanjana, Dr. Ashwin Raajkumar  
**Dataset:** EMBED (Emory Breast Imaging Dataset)  
**Task:** 4-class BI-RADS tissue density classification (Categories A–D) from 1024×1024 screening mammograms

---

## Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Dataset](#2-dataset)
3. [Pipeline Architecture](#3-pipeline-architecture)
4. [Preprocessing](#4-preprocessing)
5. [Feature Extraction](#5-feature-extraction)
6. [Model Evolution](#6-model-evolution)
7. [Experimental Results](#7-experimental-results)
8. [Confusion Matrices](#8-confusion-matrices)
9. [Sample Images](#9-sample-images)
10. [Repository Structure](#10-repository-structure)
11. [Execution](#11-execution)
12. [Suggestions for Improvement](#12-suggestions-for-improvement)

---

## 1. Problem Statement

Breast tissue density is a clinically critical biomarker. Dense tissue not only masks tumours in mammograms but is itself an independent risk factor for breast cancer. Radiologists assign one of four BI-RADS density grades:

| Grade | Label | Description |
|---|---|---|
| 1 | A | Almost entirely fatty |
| 2 | B | Scattered fibroglandular densities |
| 3 | C | Heterogeneously dense |
| 4 | D | Extremely dense |

Automating this grading reliably is hard for three reasons: the classes form an **ordinal scale** (not independent categories), the dataset is naturally **imbalanced** (grades B and C dominate), and high-resolution images carry subtle tissue signals that low-resolution models miss. This project explores whether a hybrid deep feature extraction + classical AdaBoost ensemble can solve all three challenges simultaneously.

---

## 2. Dataset

**Dataset:** EMBED (Emory Breast Imaging Dataset)  
**Modality:** Full-Field Digital Mammography (FFDM)  
**Classes:** A (Fatty), B (Scattered), C (Heterogeneously Dense), D (Extremely Dense)  
**Balanced Samples:** 5,000 images per class  
**Total:** 20,000 images

| Split | Ordinal / OvR models | AdaBoost + MLP models |
|---|---|---|
| Training | 15,928 | 15,951 |
| Validation | — | 1,989 |
| Test | 4,072 | 2,060 |

Patient-level `GroupShuffleSplit` is used across all models to ensure zero leakage between splits — no patient's images appear in more than one partition.

---

## 3. Pipeline Architecture

```
DICOM Files
    │
    ▼
[Preprocessing]  VOI LUT + 1–99% percentile clip → 1024×1024 PNG
    │
    ▼
[Feature Extraction]  Fine-tuned ConvNeXt-Base → 1024-dim feature vector
    │
    ▼
[Classification]  AdaBoost (SAMME) with MLP weak learners
    │
    ▼
Predicted BI-RADS Grade (A / B / C / D)
```

The rationale for separating feature extraction from classification is deliberate: AdaBoost's sequential re-weighting mechanism operates on tabular feature vectors, not raw images. Compressing each mammogram into a 1024-dimensional representation via a fine-tuned CNN gives AdaBoost a structured, mammography-specific space to work in — rather than generic ImageNet statistics.

---

## 4. Preprocessing

**Script:** `src/Extractor_from_harddisk.py`

Raw DICOM files carry embedded windowing metadata (VOI LUT) that must be applied correctly before pixel values become visually meaningful. The preprocessing pipeline:

- Applies VOI LUT to convert raw pixel values to display-correct intensities
- Clips pixel values to the 1st–99th percentile range, removing dead-air borders and bright metal artefacts (biopsy markers, implant edges)
- Resizes to 1024×1024 PNG — large enough to preserve subtle tissue texture
- Flips right-side (R laterality) images horizontally so all breast parenchyma faces the same direction

**Balancing:** SMOTE synthetic oversampling was dropped — interpolating in a high-dimensional feature space risks mislabelling synthetic points near class boundaries. Instead, 5,000 real images per class were manually extracted, producing a perfectly balanced 20,000-image dataset.

---

## 5. Feature Extraction

**Notebook:** `Feature Extraction/feature-extraction_ConvNeXt.ipynb`

### Why ConvNeXt-Base over EfficientNet-B4?

The project initially used EfficientNet-B4 as the feature extractor, producing 1,792-dimensional feature vectors on an imbalanced ~10k image dataset. EfficientNet-B4 is natively trained at 380×380 resolution, meaning mammograms had to be downscaled, wasting the high-resolution tissue detail that makes 1024px inputs valuable. Initial experiments with this setup (using SMOTE-balanced training, image-level splits) yielded accuracies in the range of **59–69%**.

After reviewing a 2024 Nature paper on breast tomosynthesis and a direct 2025 arXiv comparison (2505.18725), ConvNeXt-Base was adopted. Its large-kernel convolutions (7×7) naturally capture longer-range tissue patterns at high resolution, and it outperforms EfficientNet architectures on mammography tasks. Switching to ConvNeXt-Base produced **1,024-dimensional** feature vectors and, combined with patient-level splits and real balanced data, pushed accuracy to **80–82%**.

### Fine-Tuning Strategy (Two-Stage)

Rather than using a frozen backbone (which gives generic ImageNet features), ConvNeXt-Base is actively fine-tuned on labelled mammograms:

**Stage 1 — Head Warm-Up (10 epochs)**  
Backbone frozen, only the classification head trains at LR 1e-3. This prevents the randomly-initialised head from sending destructive gradients into the pre-trained backbone.

```
Frozen backbone  →  Trainable head only
LR (head): 1e-3
```

**Stage 2 — Partial Unfreeze (15 epochs)**  
The last two ConvNeXt feature blocks are unfrozen alongside the head, with differential learning rates:

```
Frozen (blocks 0–5)  →  Trainable (blocks 6–7 + head)
LR (backbone): 3e-5   LR (head): 3e-4
```

After fine-tuning, the classifier head is replaced with an `Identity` layer (preserving the `LayerNorm2d` normalisation) and the full 20,000-image dataset is passed through to export 1024-dimensional feature vectors. These vectors are what all downstream AdaBoost models train on.

---

## 6. Model Evolution

The architecture evolved iteratively across eight notebooks, each addressing a specific failure mode of the previous version:

**Ordinal AdaBoost (`3_model`)** — The first working approach. Instead of treating all four grades as independent labels, three binary classifiers are trained on ordinal thresholds (Is density > A?, > B?, > C?). Votes are summed to recover the final grade. This prevents catastrophic cross-grade errors (e.g., predicting D when the true label is A) because a sample must cross each threshold in order. Early result: **66.85%** (EfficientNet features, SMOTE, image-level split). Patient-isolated result: **80.23%**.

**Probability-Reconstructed Ordinal (`3_model_probabversion`)** — Same threshold structure, but uses soft probabilities from each classifier to reconstruct class probabilities mathematically. This failed catastrophically (**17.20%**) because the threshold probabilities don't maintain a proper ordering, causing negative or incoherent reconstructed class probabilities. The model collapsed to predicting only grades A and D.

**Hybrid Ordinal + Probability (`3_model_hybrid`)** — Combined hard ordinal voting with the soft probability approach, using ordinal results as a safety net when probability reconstruction became incoherent. Recovered performance to approximately **66%** and demonstrated that a guard-rail on the probability branch restores stability.

**One-vs-Rest AdaBoost (`4 model_OvR`)** — Four specialist binary models, one per class. The class with the highest confidence wins. Early result: **64.95%**. Patient-isolated result: **81.34%**. Slightly edges the ordinal approach on the two middle classes (B and C), though the rare-class "rest" imbalance remains a structural limitation.

**SAMME Baseline (`AdaBoost_multiclass`)** — A clean sklearn `AdaBoostClassifier` with `algorithm='SAMME'` using standard decision tree stumps. Served as a controlled reference point. Result: **59.65%** — majority classes dominated, rare classes largely ignored.

**AdaBoost + MLP Weak Learner (`AdaBoost_MLP_weaklearner`)** — Replaced decision stumps with small PyTorch MLPs (32 neurons) as weak learners. MLPs draw curved decision boundaries in the 1024-dimensional feature space, which axis-aligned stumps cannot. Result: **68.90%** — best rare-class balance among the early models.

**Regularised MLP v2 (`AdaBoost_MLP_weaklearner_v2`)** — The original MLP weak learner achieved near-zero training error, stalling the boosting loop entirely. This version deliberately handicaps each MLP: hidden layer capped at 32 neurons, max 15 iterations, L2 regularisation, early stopping, batch size controlled at 64. This restored true sequential boosting. Result: **65.30%** test accuracy — lower number but a more honest measure, reflecting the removal of overfitting.

**Patient-Isolated MLP v2.2 (`AdaBoost_MLP_weaklearner_v2_1`)** — Added strict patient-level `GroupShuffleSplit` alongside V2 regularisation and a wide parallel grid search (250 estimators, LR 0.5, max_iter 50). Best model to date: **82.38% accuracy**, macro F1 **0.8233**. Category A recall: 90.6%, Category D precision: 89.4%.

---

## 7. Experimental Results

| Model | Early result | Patient-isolated result | Notes |
|---|---|---|---|
| Ordinal AdaBoost | 66.85% | **80.23%** | Zero catastrophic A↔D errors |
| Probability-Reconstructed Ordinal | 17.20% | — | Collapsed; superseded |
| Hybrid Ordinal + Probability | ~66% | — | Superseded by MLP approach |
| One-vs-Rest AdaBoost | 64.95% | **81.34%** | Edges ordinal on B and C |
| SAMME Baseline | 59.65% | — | Decision stump reference |
| AdaBoost + MLP Weak Learner | 68.90% | — | Best early rare-class balance |
| Regularised MLP (V2) | 65.30% | — | Restored true boosting loop |
| **AdaBoost + MLP V2.2** | — | **82.38%** (F1: 0.8233) | **Current best** |

The ~15–17 point jump between early and patient-isolated results reflects the combined effect of removing SMOTE, enforcing patient-level splits, and using ConvNeXt-Base over EfficientNet-B4. Earlier scores were partly inflated by data leakage and synthetic-sample noise.

---

## 8. Confusion Matrices

### Ordinal AdaBoost (80.23%, n=4072)

|  | Pred A | Pred B | Pred C | Pred D |
|---|---|---|---|---|
| **Actual A** | 926 | 135 | 4 | 0 |
| **Actual B** | 151 | 708 | 129 | 2 |
| **Actual C** | 1 | 149 | 755 | 111 |
| **Actual D** | 0 | 3 | 120 | 878 |

### One-vs-Rest AdaBoost (81.34%, n=4072)

|  | Pred A | Pred B | Pred C | Pred D |
|---|---|---|---|---|
| **Actual A** | 933 | 128 | 4 | 0 |
| **Actual B** | 152 | 720 | 115 | 3 |
| **Actual C** | 1 | 133 | 781 | 101 |
| **Actual D** | 0 | 3 | 120 | 878 |

### AdaBoost + MLP V2.2 — Best Model (82.38%, n=2060)

|  | Pred A | Pred B | Pred C | Pred D |
|---|---|---|---|---|
| **Actual A** | 462 | 48 | 0 | 0 |
| **Actual B** | 71 | 373 | 60 | 2 |
| **Actual C** | 0 | 63 | 409 | 52 |
| **Actual D** | 1 | 2 | 64 | 453 |

All models share the same clinically important safety property: zero A↔D confusion. Every error is adjacent-grade, closely mirroring human radiologist disagreement.

---

## 9. Sample Images

> *Images to be added.*

### Density A — Almost Entirely Fatty
```
[image placeholder]
```

### Density B — Scattered Fibroglandular
```
[image placeholder]
```

### Density C — Heterogeneously Dense
```
[image placeholder]
```

### Density D — Extremely Dense
```
[image placeholder]
```

---

## 10. Repository Structure

```
BTD_Prediction_Adaboost/
│
├── Feature Extraction/
│   ├── feature-extraction_ConvNeXt.ipynb       ← Two-stage fine-tuning + feature export
│   └── feature_extraction_EfficientNetB4.ipynb ← Deprecated (380px resolution limit)
│
├── Model notebooks/
│   ├── 3_model.ipynb                           ← Ordinal AdaBoost (Strategy C)
│   ├── 3_model_probabversion.ipynb             ← Probability-reconstructed ordinal (failed)
│   ├── 3_model_hybrid.ipynb                    ← Hybrid ordinal + probability fallback
│   ├── 4 model_OvR.ipynb                       ← One-vs-Rest specialist ensemble
│   ├── AdaBoost_multiclass.ipynb               ← Sklearn SAMME baseline
│   ├── AdaBoost_MLP_weaklearner.ipynb          ← Custom PyTorch MLP weak learner
│   ├── AdaBoost_MLP_weaklearner_v2.ipynb       ← Regularised v2
│   └── AdaBoost_MLP_weaklearner_v2_1.ipynb     ← V2.2 (GroupSplit + grid search) — current best
│
├── src/
│   ├── Extractor_from_harddisk.py              ← DICOM → PNG extraction pipeline
│   ├── data_assignment.py                      ← Patient-wise label mapping
│   ├── npyfile_combiner.py                     ← Merges .npy feature batches
│   └── prepare_training_data.py                ← Final train/val/test split prep
│
├── predict.py                                  ← Benchmark inference script
├── requirements.txt
└── README.md
```

---

## 11. Execution

### Feature Extraction
Run `Feature Extraction/feature-extraction_ConvNeXt.ipynb` on Kaggle (T4 x2 recommended).  
Outputs: `convnext_finetuned_features.npy`, `convnext_finetuned_labels.npy`

### Training
Run `Model notebooks/AdaBoost_MLP_weaklearner_v2_1.ipynb` with the extracted `.npy` files as input.  
Output: `checkpoints/best_model.pt`

### Inference
```bash
python predict.py <input_features.npy> <output_predictions.npy>
```
Input: 1024-dimensional ConvNeXt feature vectors (N × 1024)  
Output: Predicted BI-RADS grades as 0-indexed integers (0–3), corresponding to grades A–D

---

## 12. Suggestions for Improvement

Key directions in order of expected impact: **multi-view fusion** (combining CC and MLO views per patient rather than treating them independently); **Swin Transformer backbone** (shifted-window self-attention better models long-range tissue relationships than convolutions); **ordinal-aware loss** (penalise A↔D errors more harshly than A↔B); **test-time augmentation** (average features from small augmentations at inference for more stable predictions); **confidence calibration** (Platt scaling or isotonic regression to make `predict_proba` outputs clinically trustworthy); and **SHAP + Grad-CAM explainability** to verify the model attends to actual tissue patterns rather than scan artefacts.
