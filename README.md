# BTD_Prediction_Adaboost

## Breast Tissue Density (BI-RADS) Classification via AdaBoost + MLP Ensemble

**Author:** Kaligatla Devi Prasad

**Dataset:** EMBED (Emory Breast Imaging Dataset)

**Task:** 4-class BI-RADS tissue density classification (Categories 1–4) from 1024×1024 screening mammograms

---

## Table of Contents

1. [Problem Statement](https://www.google.com/search?q=%231-problem-statement)
2. [Pipeline Architecture](https://www.google.com/search?q=%232-pipeline-architecture)
3. [Preprocessing](https://www.google.com/search?q=%233-preprocessing)
4. [Feature Extraction](https://www.google.com/search?q=%234-feature-extraction)
5. [Model Evolution](https://www.google.com/search?q=%235-model-evolution)
6. [Experimental Results](https://www.google.com/search?q=%236-experimental-results)
7. [Repository Structure](https://www.google.com/search?q=%237-repository-structure)
8. [Execution](https://www.google.com/search?q=%238-execution)
9. [Suggestions for Improvement](https://www.google.com/search?q=%239-suggestions-for-improvement)

---

## 1. Problem Statement

Breast tissue density is a clinically critical biomarker. Dense tissue not only masks tumours in mammograms but is itself an independent risk factor for breast cancer. Radiologists assign one of four BI-RADS density grades:

| Grade | Description |
| --- | --- |
| 1 | Almost entirely fatty |
| 2 | Scattered fibroglandular densities |
| 3 | Heterogeneously dense |
| 4 | Extremely dense |

Automating this grading reliably is hard for three reasons: the classes form an **ordinal scale** (not independent categories), the dataset is naturally **imbalanced** (grades 2 and 3 dominate), and high-resolution images carry subtle tissue signals that low-resolution models miss. This project explores whether a hybrid deep feature extraction + classical AdaBoost ensemble can solve all three challenges simultaneously.

---

## 2. Pipeline Architecture

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
Predicted BI-RADS Grade (0–3 internally, reported as 1–4)

```

The rationale for splitting feature extraction from classification (rather than end-to-end training) is deliberate: AdaBoost's sequential re-weighting mechanism operates on tabular feature vectors, not raw images. By first compressing each mammogram into a rich 1024-dimensional representation via a fine-tuned CNN, we give AdaBoost a structured, information-dense space to work in — one that captures mammography-specific tissue patterns rather than generic ImageNet statistics.

---

## 3. Preprocessing

**Script:** `src/Extractor_from_harddisk.py`

Raw DICOM files carry embedded windowing metadata (VOI LUT) that must be applied correctly before pixel values become visually meaningful. The preprocessing pipeline:

* Applies VOI LUT to convert raw pixel values to display-correct intensities
* Clips pixel values to the 1st–99th percentile range, removing dead-air borders and bright metal artefacts (biopsy markers, implant edges) that would otherwise dominate the CNN's attention
* Resizes to 1024×1024 PNG — large enough to preserve subtle tissue texture
* Flips right-side (R laterality) images horizontally so all breast parenchyma faces the same direction, removing orientation as a confounding variable

**Balancing:** Rather than using SMOTE synthetic oversampling (which interpolates in a 1024-dimensional space and risks mislabelling synthetic points near class boundaries), 5,000 images were manually extracted per class, producing a perfectly balanced 20,000-image dataset.

---

## 4. Feature Extraction

**Notebook:** `Feature Extraction/feature-extraction_ConvNeXt.ipynb`

### Why ConvNeXt-Base over EfficientNet-B4?

EfficientNet-B4 was the original extractor, but it is natively trained at 380×380 resolution. Upscaling mammograms to feed it wastes the high-resolution tissue detail that makes 1024px inputs valuable in the first place. A 2024 Nature paper on breast tomosynthesis and a direct 2025 arXiv comparison (2505.18725) both confirm that ConvNeXt-Base outperforms EfficientNet architectures on mammography tasks, largely because its large-kernel convolutions (7×7) naturally capture longer-range tissue patterns at high resolution.

### Fine-Tuning Strategy (Two-Stage)

Rather than using the backbone in a frozen state (which gives generic ImageNet features), the ConvNeXt-Base is actively fine-tuned on labelled mammograms in two sequential stages:

**Stage 1 — Head Warm-Up (10 epochs)** The entire ConvNeXt backbone is frozen. Only the classification head trains. This prevents the randomly-initialised head from sending destructive gradients into the pre-trained backbone before it has stabilised.

```
Frozen backbone  →  Trainable head only
LR (head): 1e-3

```

**Stage 2 — Partial Unfreeze (15 epochs)** The last two ConvNeXt feature blocks are unfrozen alongside the head. Differential learning rates prevent the backbone from changing too aggressively:

```
Frozen (blocks 0–5)  →  Trainable (blocks 6–7 + head)
LR (backbone): 3e-5   LR (head): 3e-4

```

After fine-tuning, the classifier head is replaced with a `Flatten` layer, and the entire 20,000-image dataset is passed through to extract 1024-dimensional feature vectors. These vectors — not raw images — are what all downstream AdaBoost models train on.

---

## 5. Model Evolution

All model notebooks are in `/Model notebooks`. The architecture evolved iteratively, with each version addressing a specific failure mode of the previous one.

---

### 5.1 `3_model.ipynb` — Initial Multiclass AdaBoost

**Logic:** The first exploratory attempt at standard multiclass AdaBoost.
**Why it struggled:** Suffered from the natural class imbalance and lacked ordinal awareness, treating all four density grades as independent labels.

---

### 5.2 `3_model_probabversion.ipynb` — Probability-Based Decoding

**Logic:** Instead of hard classes, this approach used direct probability-based decoding to reconstruct predictions based on confidence scores.
**Why it underperformed:** The probability reconstruction method proved highly unreliable for final prediction logic. The soft probabilities across classes were not confident enough to establish clear decision boundaries.
**Result:** 17.20% accuracy.

---

### 5.3 `3_model_hybrid.ipynb` — Hybrid Ordinal Decoding

**Logic:** Because the probability decoder was unstable, this introduced a hybrid system using sequential thresholds (Density > Grade 1, > Grade 2, etc.). If the probability branch failed to reach consensus, the model fell back to soft voting across the ordinal thresholds.
**Why it improved:** The ordinal structure eliminated catastrophic cross-grade errors (like predicting a 4 when the true label is 1) because a sample must cross each boundary in order.
**Result:** 66.85% accuracy (on successful fallback runs). Near-zero catastrophic misclassifications.

---

### 5.4 `4 model_OvR.ipynb` — One-vs-Rest AdaBoost

**Logic:** Four specialist AdaBoost models are trained, one per class. Each model answers a single binary question: "Is this image class X, or not?" The class whose specialist fires with the highest confidence wins.
**Why it underperformed:** Each specialist's "rest" group is 3× larger than its positive class. For the rare grades (1 and 4), this imbalance is severe — the "rest" pool is enormous and the majority classes mathematically crowd out the minority ones.
**Result:** 64.95% overall accuracy, and worse rare-class detection than the ordinal approach.

---

### 5.5 `AdaBoost_multiclass.ipynb` — Sklearn SAMME AdaBoost (Baseline)

**Logic:** A clean sklearn `AdaBoostClassifier` with `algorithm='SAMME'` using standard decision stumps. SAMME (Stagewise Additive Modelling using a Multi-class Exponential loss) is the correct multiclass extension of AdaBoost. This notebook served as the formal baseline to benchmark exactly how much the custom MLP weak learners add over standard decision stumps.
**Why it struggled:** The model learned to optimise for the majority classes and largely ignored grades 1 and 4.
**Result:** 59.65% accuracy.

---

### 5.6 `AdaBoost_MLP_weaklearner.ipynb` — AdaBoost + PyTorch MLP (Architecture B)

**Logic:** Decision tree stumps are extremely limited learners — each one partitions the 1024-dimensional feature space with a single axis-aligned cut. This notebook replaces them with small PyTorch MLPs as the weak learners (32 neurons), giving it just enough capacity to learn non-linear feature interactions.
**Why it improved:** MLPs can draw curved decision boundaries in feature space, which axis-aligned tree stumps cannot.
**Result:** 68.90% accuracy. Best rare-class balance of the image-split models.

---

### 5.7 `AdaBoost_MLP_weaklearner_v2.ipynb` — Regularised AdaBoost + MLP (V2)

**Logic:** The previous MLP weak learner had a critical flaw — it was too powerful, achieving near-zero training error and stalling the boosting loop. This version deliberately handicaps each MLP by adding L2 regularisation (`alpha=1e-3`), early stopping with an internal validation split, and batch-size control.
**Result:** 69.00% validation accuracy, 65.30% test accuracy. The gap between validation and test is healthy — prior versions that showed identical scores were overfitting silently.

---

### 5.8 `AdaBoost_MLP_weaklearner_v2_1.ipynb` — Patient-Level GroupSplit (Current)

**Logic:** Previous models utilized a random image-level split. Upon deeper data analysis, it was discovered that the dataset contained multiple images per patient. A random split inevitably leaked images from the same patient into both the training and test sets, artificially inflating accuracy as the model memorized patient-specific anatomical quirks rather than generalized tissue-density rules.
This finalized version introduces strict patient-level `GroupShuffleSplit` (zero data leakage) alongside the V2 regularisation.
**Result:** `[RESULTS PENDING - AWAITING CONVNEXT FEATURE EXTRACTION]`

---

## 6. Experimental Results

*Note: Models prior to v2.1 utilized image-level splitting which contained patient overlap. V2.1 introduces strict patient isolation.*

| Model | Notebook | Accuracy | Notes |
| --- | --- | --- | --- |
| Initial Multiclass | `3_model.ipynb` | - | Exploratory baseline run |
| Probability Decoder | `3_model_probabversion.ipynb` | 17.20% | Confidence reconstruction failed |
| Hybrid / Ordinal | `3_model_hybrid.ipynb` | 66.85% | Ordinal fallback eliminated catastrophic errors |
| One-vs-Rest AdaBoost | `4 model_OvR.ipynb` | 64.95% | Rare classes drowned out by majority "rest" |
| SAMME Baseline | `AdaBoost_multiclass.ipynb` | 59.65% | Standard SAMME with decision stumps |
| AdaBoost + MLP (Arc B) | `AdaBoost_MLP_weaklearner.ipynb` | 68.90% | Best rare-class balance prior to group splitting |
| Regularised MLP (V2) | `AdaBoost_MLP_weaklearner_v2.ipynb` | 65.30% (test) | Added validation split and regularisation |
| **MLP GroupSplit (V2.1)** | `AdaBoost_MLP_weaklearner_v2_1.ipynb` | **[PENDING]** | **Strict patient-level isolation (Zero Leakage)** |

### Current Best — V2.1 Test Confusion Matrix

```text
[AWAITING FINAL TRAINING RUN ON NEW CONVNEXT FEATURES]

```

---

## 7. Repository Structure

```
BTD_Prediction_Adaboost/
│
├── Feature Extraction/
│   ├── feature-extraction_ConvNeXt.ipynb       ← Two-stage fine-tuning + feature export
│   └── feature_extraction_EfficientNetB4.ipynb ← Deprecated (380px resolution limit)
│
├── Model notebooks/
│   ├── 3_model.ipynb                           ← Initial exploratory multiclass AdaBoost
│   ├── 3_model_probabversion.ipynb             ← Probabilistic decoding variant
│   ├── 3_model_hybrid.ipynb                    ← Ordinal threshold / hybrid fallback
│   ├── 4 model_OvR.ipynb                       ← One-vs-Rest specialist ensemble
│   ├── AdaBoost_multiclass.ipynb               ← Sklearn SAMME reference baseline
│   ├── AdaBoost_MLP_weaklearner.ipynb          ← Custom PyTorch MLP weak learner
│   ├── AdaBoost_MLP_weaklearner_v2.ipynb       ← Regularised v2
│   └── AdaBoost_MLP_weaklearner_v2_1.ipynb     ← Regularised v2.1 (GroupSplit / Zero Leakage)
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

## 8. Execution

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

Output: Predicted BI-RADS grades as 0-indexed integers (0–3), corresponding to grades 1–4

---

## 9. Suggestions for Improvement

These are the most impactful next steps roughly in order of expected gain:

**1. Multi-View Fusion** Each patient has both CC (cranio-caudal) and MLO (mediolateral oblique) views. Currently each view is treated as an independent sample. A multi-view model that concatenates or attention-pools features from both views of the same breast before classification would give the AdaBoost model richer, more complete information — this is the approach taken by the top-performing MV-Swin-T (2024) paper.

**2. Swin Transformer Backbone** ConvNeXt is an excellent CNN, but Swin Transformers have a structural advantage for high-resolution mammography: their shifted-window self-attention can model long-range spatial dependencies (e.g., the relationship between tissue in one quadrant and another) that convolutions approximate only locally. A 2024 Nature paper specifically showed resolution-scaling benefits of Swin for breast tomosynthesis beyond CNN architectures.

**3. Label Smoothing** Since the classes are ordinal, a misclassification of grade 2 as grade 3 should be penalised less than grade 2 as grade 4. Standard cross-entropy treats all errors equally. Replacing it with an ordinal-aware loss (e.g., Earth Mover's Distance loss or a cost-sensitive cross-entropy with an ordinal penalty matrix) would directly encode clinical severity into training.

**4. Test-Time Augmentation (TTA)** At inference time, each image could be passed through the ConvNeXt extractor multiple times with slight augmentations (horizontal flip, small rotations, brightness shifts), and the resulting feature vectors averaged before being fed to AdaBoost. This reduces variance in the extracted features at zero additional training cost.

**5. Confidence Calibration** AdaBoost's raw `predict_proba` outputs are not well-calibrated probabilities. Applying Platt scaling or isotonic regression (via sklearn's `CalibratedClassifierCV`) after training would make the confidence scores more trustworthy for downstream clinical decision support.

**6. Ensemble the AdaBoost Variants** Methods A (Ordinal), B (OvR), and the current MLP v2.1 each have different failure modes. A soft-vote ensemble of all three — using their `predict_proba` outputs — may outperform any individual model, since their errors are partially uncorrelated.

**7. Explainability Layer** Adding SHAP (SHapley Additive exPlanations) analysis on the AdaBoost model's feature importances, mapped back to spatial regions of the ConvNeXt feature map via Grad-CAM, would allow visual verification that the model is attending to actual tissue patterns rather than artefacts or image borders.
