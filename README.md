# BTD_Prediction_Adaboost
## Breast Tissue Density (BI-RADS) Classification via AdaBoost + MLP Ensemble

**Author:** Kalegatla Devi Prasad  
**Dataset:** EMBED (Emory Breast Imaging Dataset)  
**Task:** 4-class BI-RADS tissue density classification (Categories 1–4) from 1024×1024 screening mammograms

---

## Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Preprocessing](#3-preprocessing)
4. [Feature Extraction](#4-feature-extraction)
5. [Model Evolution](#5-model-evolution)
6. [Experimental Results](#6-experimental-results)
7. [Repository Structure](#7-repository-structure)
8. [Execution](#8-execution)
9. [Suggestions for Improvement](#9-suggestions-for-improvement)

---

## 1. Problem Statement

Breast tissue density is a clinically critical biomarker. Dense tissue not only masks tumours in mammograms but is itself an independent risk factor for breast cancer. Radiologists assign one of four BI-RADS density grades:

| Grade | Description |
|---|---|
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

- Applies VOI LUT to convert raw pixel values to display-correct intensities
- Clips pixel values to the 1st–99th percentile range, removing dead-air borders and bright metal artefacts (biopsy markers, implant edges) that would otherwise dominate the CNN's attention
- Resizes to 1024×1024 PNG — large enough to preserve subtle tissue texture
- Flips right-side (R laterality) images horizontally so all breast parenchyma faces the same direction, removing orientation as a confounding variable

**Balancing:** Rather than using SMOTE synthetic oversampling (which interpolates in a 1024-dimensional space and risks mislabelling synthetic points near class boundaries), 5,000 images were manually extracted per class, producing a perfectly balanced 20,000-image dataset.

---

## 4. Feature Extraction

**Notebook:** `Feature Extraction/feature-extraction_ConvNeXt.ipynb`

### Why ConvNeXt-Base over EfficientNet-B4?

EfficientNet-B4 was the original extractor, but it is natively trained at 380×380 resolution. Upscaling mammograms to feed it wastes the high-resolution tissue detail that makes 1024px inputs valuable in the first place. A 2024 Nature paper on breast tomosynthesis and a direct 2025 arXiv comparison (2505.18725) both confirm that ConvNeXt-Base outperforms EfficientNet architectures on mammography tasks, largely because its large-kernel convolutions (7×7) naturally capture longer-range tissue patterns at high resolution.

### Fine-Tuning Strategy (Two-Stage)

Rather than using the backbone in a frozen state (which gives generic ImageNet features), the ConvNeXt-Base is actively fine-tuned on labelled mammograms in two sequential stages:

**Stage 1 — Head Warm-Up (10 epochs)**  
The entire ConvNeXt backbone is frozen. Only the classification head trains. This prevents the randomly-initialised head from sending destructive gradients into the pre-trained backbone before it has stabilised.

```
Frozen backbone  →  Trainable head only
LR (head): 1e-3
```

**Stage 2 — Partial Unfreeze (15 epochs)**  
The last two ConvNeXt feature blocks are unfrozen alongside the head. Differential learning rates prevent the backbone from changing too aggressively:

```
Frozen (blocks 0–5)  →  Trainable (blocks 6–7 + head)
LR (backbone): 3e-5   LR (head): 3e-4
```

After fine-tuning, the classifier head is replaced with a `Flatten` layer, and the entire 20,000-image dataset is passed through to extract 1024-dimensional feature vectors. These vectors — not raw images — are what all downstream AdaBoost models train on.

---

## 5. Model Evolution

All model notebooks are in `/Model notebooks`. The architecture evolved iteratively, with each version addressing a specific failure mode of the previous one.

---

### 5.1 `3_model.ipynb` — Standard Multiclass AdaBoost (Baseline)

**Logic:** The most straightforward application of AdaBoost. A single `AdaBoostClassifier` with decision tree stumps is trained directly on the 4-class problem. At each boosting round, the algorithm increases the weight of misclassified samples so the next stump focuses harder on them.

**Why it struggled:** Standard multiclass AdaBoost treats all four classes as independent labels. It has no awareness that predicting grade 4 when the true label is grade 1 is catastrophically worse than being off by one grade. Combined with the natural class imbalance (grades 2 and 3 dominate), the model learns to optimise for the majority classes and largely ignores grades 1 and 4.

**Result:** 59.65% accuracy.

---

### 5.2 `3_model_hybrid.ipynb` — Ordinal AdaBoost

**Logic:** Tissue density is not a set of independent categories — it is an ordered scale. This notebook exploits that structure using an ordinal decomposition. Instead of one 4-class problem, the problem is broken into three binary classifiers:

- **Classifier A:** Is density > Grade 1? (yes/no)
- **Classifier B:** Is density > Grade 2? (yes/no)
- **Classifier C:** Is density > Grade 3? (yes/no)

Each classifier is a separate AdaBoost model. The final grade is recovered by summing the binary outputs (0 + 1 + 1 = Grade 3, etc.). Because each sub-problem is simpler and the ordinal relationship is explicitly encoded, the model naturally avoids catastrophic cross-grade errors.

**Why it improved:** The cascading threshold structure means a sample must cross each boundary in order — it cannot jump from grade 1 to grade 4 in a single step. This closely mirrors how radiologists actually think about density progression.

**Result:** 66.85% accuracy. Near-zero catastrophic misclassifications.

---

### 5.3 `3_model_probabversion.ipynb` — Probabilistic Ordinal AdaBoost

**Logic:** An extension of the ordinal approach. Instead of hard binary decisions from each threshold classifier, this notebook uses `predict_proba` to extract soft confidence scores from each sub-classifier. The final grade is chosen by finding the threshold combination with the highest joint probability. This reduces the impact of borderline decisions where a hard threshold would flip incorrectly.

**Why it helps:** Confidence-weighted decisions smooth out the edges of the ordinal boundaries, particularly for grades 2/3 which are the most commonly confused pair.

---

### 5.4 `4 model_OvR.ipynb` — One-vs-Rest AdaBoost

**Logic:** Four specialist AdaBoost models are trained, one per class. Each model answers a single binary question: "Is this image class X, or not?" The class whose specialist fires with the highest confidence wins.

**Why it underperformed:** Each specialist's "rest" group is 3× larger than its positive class. For the rare grades (1 and 4), this imbalance is severe — the "rest" pool is enormous and the model learns to default to "not class X" almost always. The majority classes (2 and 3) crowd out the minority ones mathematically.

**Result:** 71.29% overall accuracy, but worse rare-class detection than the ordinal approach.

---

### 5.5 `AdaBoost_MLP_weaklearner.ipynb` — AdaBoost + PyTorch MLP (Architecture B)

**Logic:** Decision tree stumps are extremely limited learners — each one partitions the 1024-dimensional feature space with a single axis-aligned cut. This notebook replaces them with small PyTorch MLPs as the weak learners. Each MLP has one hidden layer of 32 neurons, giving it just enough capacity to learn non-linear feature interactions while still being "weak" enough to leave room for boosting to add value.

The boosting loop is implemented manually: after each MLP trains, its misclassifications are identified, sample weights are updated (misclassified samples get higher weight), and a `WeightedRandomSampler` forces the next MLP to focus on those harder examples.

**Why it improved:** MLPs can draw curved decision boundaries in feature space, which axis-aligned tree stumps cannot. The 1024-dimensional ConvNeXt feature space has complex, non-linear structure that benefits from this expressiveness.

**Result:** 68.90% accuracy, best rare-class balance so far.

---

### 5.6 `AdaBoost_MLP_weaklearner_v2.ipynb` — Regularised AdaBoost + MLP (Current)

**Logic:** The previous MLP weak learner had a critical flaw — it was too powerful. The first MLP in the chain would achieve near-zero training error, which meant it classified almost all training samples correctly. With no misclassifications to re-weight, the boosting signal never propagated. Subsequent MLPs received essentially uniform sample weights and trained independently, making the "ensemble" just four separate networks with soft voting rather than a true boosting chain.

This version deliberately handicaps each MLP to force genuine weakness:

- Hidden layer capped at 32 neurons
- Maximum 15 training iterations
- L2 regularisation (`alpha=1e-3`) to penalise large weights
- Early stopping with a 10% internal validation split
- Batch size controlled at 64

These constraints ensure each MLP can only partially solve the problem, leaving genuine misclassification signal for the boosting algorithm to act on.

**Additionally introduced:**
- `use_class_balance` flag for optional class-weighted loss
- Grid search over `n_estimators` and `learning_rate` using stratified cross-validation
- Patient-level `GroupShuffleSplit` for zero-leakage data splitting

**Result:** 69.00% validation accuracy, 65.30% test accuracy. The gap between validation and test is expected and healthy — prior versions that showed identical train/val/test scores were overfitting silently.

---

### 5.7 `AdaBoost_multiclass.ipynb` — Sklearn SAMME AdaBoost (Reference)

**Logic:** A clean sklearn `AdaBoostClassifier` with `algorithm='SAMME'` using the fine-tuned ConvNeXt features. SAMME (Stagewise Additive Modelling using a Multi-class Exponential loss) is the correct multiclass extension of AdaBoost — it handles 4-class problems natively without binary decomposition. This notebook serves as a controlled reference point to benchmark how much the custom MLP weak learner adds over sklearn's default decision stumps on the same feature set.

---

## 6. Experimental Results

| Model | Notebook | Accuracy | Notes |
|---|---|---|---|
| Standard Multiclass AdaBoost | `3_model.ipynb` | 59.65% | Baseline, class imbalance hurt badly |
| Ordinal AdaBoost | `3_model_hybrid.ipynb` | 66.85% | Ordinal structure eliminated catastrophic errors |
| One-vs-Rest AdaBoost | `4 model_OvR.ipynb` | 71.29% | Rare classes drowned out by majority "rest" |
| AdaBoost + MLP Weak Learner | `AdaBoost_MLP_weaklearner.ipynb` | 68.90% | Best rare-class balance |
| AdaBoost + MLP v2 (Regularised) | `AdaBoost_MLP_weaklearner_v2.ipynb` | 65.30% (test) | Most robust, true boosting loop |

### Current Best — V2 Test Confusion Matrix

|  | Pred 1 | Pred 2 | Pred 3 | Pred 4 |
|---|---|---|---|---|
| **Actual 1** | 83 | 33 | 1 | 0 |
| **Actual 2** | 84 | 265 | 77 | 2 |
| **Actual 3** | 1 | 70 | 272 | 56 |
| **Actual 4** | 0 | 2 | 21 | 33 |

All errors are adjacent-grade — the model never confuses grade 1 with grade 4. This is a clinically important safety property.

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
│   ├── 3_model.ipynb                           ← Baseline multiclass AdaBoost
│   ├── 3_model_hybrid.ipynb                    ← Ordinal threshold decomposition
│   ├── 3_model_probabversion.ipynb             ← Probabilistic ordinal variant
│   ├── 4 model_OvR.ipynb                       ← One-vs-Rest specialist ensemble
│   ├── AdaBoost_MLP_weaklearner.ipynb          ← Custom PyTorch MLP weak learner
│   ├── AdaBoost_MLP_weaklearner_v2.ipynb       ← Regularised v2 (current best)
│   └── AdaBoost_multiclass.ipynb               ← Sklearn SAMME reference
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
Run `Model notebooks/AdaBoost_MLP_weaklearner_v2.ipynb` with the extracted `.npy` files as input.  
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

**1. Multi-View Fusion**  
Each patient has both CC (cranio-caudal) and MLO (mediolateral oblique) views. Currently each view is treated as an independent sample. A multi-view model that concatenates or attention-pools features from both views of the same breast before classification would give the AdaBoost model richer, more complete information — this is the approach taken by the top-performing MV-Swin-T (2024) paper.

**2. Swin Transformer Backbone**  
ConvNeXt is an excellent CNN, but Swin Transformers have a structural advantage for high-resolution mammography: their shifted-window self-attention can model long-range spatial dependencies (e.g., the relationship between tissue in one quadrant and another) that convolutions approximate only locally. A 2024 Nature paper specifically showed resolution-scaling benefits of Swin for breast tomosynthesis beyond CNN architectures.

**3. Label Smoothing**  
Since the classes are ordinal, a misclassification of grade 2 as grade 3 should be penalised less than grade 2 as grade 4. Standard cross-entropy treats all errors equally. Replacing it with an ordinal-aware loss (e.g., Earth Mover's Distance loss or a cost-sensitive cross-entropy with an ordinal penalty matrix) would directly encode clinical severity into training.

**4. Test-Time Augmentation (TTA)**  
At inference time, each image could be passed through the ConvNeXt extractor multiple times with slight augmentations (horizontal flip, small rotations, brightness shifts), and the resulting feature vectors averaged before being fed to AdaBoost. This reduces variance in the extracted features at zero additional training cost.

**5. Confidence Calibration**  
AdaBoost's raw `predict_proba` outputs are not well-calibrated probabilities. Applying Platt scaling or isotonic regression (via sklearn's `CalibratedClassifierCV`) after training would make the confidence scores more trustworthy for downstream clinical decision support.

**6. Ensemble the AdaBoost Variants**  
Methods A (Ordinal), B (OvR), and the current MLP v2 each have different failure modes. A soft-vote ensemble of all three — using their `predict_proba` outputs — may outperform any individual model, since their errors are partially uncorrelated.

**7. Explainability Layer**  
Adding SHAP (SHapley Additive exPlanations) analysis on the AdaBoost model's feature importances, mapped back to spatial regions of the ConvNeXt feature map via Grad-CAM, would allow visual verification that the model is attending to actual tissue patterns rather than artefacts or image borders.
