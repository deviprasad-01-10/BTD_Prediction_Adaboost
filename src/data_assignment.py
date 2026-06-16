import pandas as pd
import numpy as np
import os

# ── Configuration ────────────────────────────────────────
CSV_PATH                = r"C:\Users\kalig\Downloads\embed_merged_without_blank_tissueden.csv"
FEATURES_PATH           = r"C:\Users\kalig\OneDrive\Desktop\features.npy"
FILENAMES_PATH          = r"C:\Users\kalig\OneDrive\Desktop\filenames.npy"
ALIGNED_FEATURES_OUTPUT = r"C:\Users\kalig\OneDrive\Desktop\train_X.npy"
ALIGNED_LABELS_OUTPUT   = r"C:\Users\kalig\OneDrive\Desktop\train_y.npy"
ALIGNED_GROUPS_OUTPUT   = r"C:\Users\kalig\OneDrive\Desktop\train_groups.npy"

# ── Load CSV ─────────────────────────────────────────────
print("Loading CSV metadata...")
# Load exactly the three columns we need for this pipeline
df = pd.read_csv(CSV_PATH, usecols=['anon_dicom_path', 'tissueden', 'patient_id'])

# Extract the base filename to act as our primary key
df['base_id'] = (df['anon_dicom_path'].astype(str)
                 .str.replace('\\', '/', regex=False)
                 .str.split('/').str[-1]
                 .str.replace('.dcm', '', regex=False))

# Build our two mapping dictionaries
label_dict = dict(zip(df['base_id'], df['tissueden']))
group_dict = dict(zip(df['base_id'], df['patient_id']))

print(f"✅ Built dictionaries with {len(label_dict)} records.")
print(f"   Unique tissueden values: {df['tissueden'].unique()}")
print(f"   Unique patients found: {df['patient_id'].nunique()}")

# ── Load features & filenames ────────────────────────────
print("\nLoading features and filenames from disk...")
X             = np.load(FEATURES_PATH)
raw_filenames = np.load(FILENAMES_PATH)

assert len(X) == len(raw_filenames), \
    f"Mismatch! X={len(X)} rows but filenames={len(raw_filenames)} entries."

# ── Direct mapping — every image is guaranteed a match ───
print("Mapping labels and groups to features...")
base_ids = pd.Series(raw_filenames).apply(
    lambda p: os.path.splitext(os.path.basename(p))[0]
)

# Apply both dictionaries to ensure perfect row-by-row alignment
y_final = base_ids.map(label_dict).values.astype(int)
groups_final = base_ids.map(group_dict).values

# ── Save ─────────────────────────────────────────────────
np.save(ALIGNED_FEATURES_OUTPUT, X)         
np.save(ALIGNED_LABELS_OUTPUT,   y_final)
np.save(ALIGNED_GROUPS_OUTPUT,   groups_final)

# ── Report ────────────────────────────────────────────────
unique, counts = np.unique(y_final, return_counts=True)

print("\n=== ALIGNMENT COMPLETE ===")
print(f"Total images      : {len(y_final)}")
print(f"Final X shape     : {X.shape}")
print(f"Final y shape     : {y_final.shape}")
print(f"Final group shape : {groups_final.shape}")
print(f"Class distribution: {dict(zip(unique, counts))}")
print(f"Saved to          : {ALIGNED_FEATURES_OUTPUT}")
print(f"               and: {ALIGNED_LABELS_OUTPUT}")
print(f"               and: {ALIGNED_GROUPS_OUTPUT}")