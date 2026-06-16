import numpy as np

# 1. Load Old Data
X_old = np.load(r"C:\...\train_X.npy")
y_old = np.load(r"C:\...\train_y.npy")
groups_old = np.load(r"C:\...\train_groups.npy")  # <-- ADDED

# 2. Load New Data
X_new = np.load(r"C:\...\train_X_2.npy")
y_new = np.load(r"C:\...\train_y_2.npy")
groups_new = np.load(r"C:\...\train_groups_2.npy") # <-- ADDED

# 3. Combine All Three
X_combined = np.vstack([X_old, X_new])
y_combined = np.concatenate([y_old, y_new])
groups_combined = np.concatenate([groups_old, groups_new]) # <-- ADDED

# 4. Save the Unified Dataset
np.save(r"C:\...\train_X_combined.npy", X_combined)
np.save(r"C:\...\train_y_combined.npy", y_combined)
np.save(r"C:\...\train_groups_combined.npy", groups_combined) # <-- ADDED

print("\n=== MERGE SUCCESSFUL ===")
print(f"Features : Old: {X_old.shape[0]} | New: {X_new.shape[0]} -> Combined: {X_combined.shape}")
print(f"Labels   : Combined length = {len(y_combined)}")
print(f"Groups   : Combined length = {len(groups_combined)}")

# Sanity Check
assert len(X_combined) == len(y_combined) == len(groups_combined), "CRITICAL: Arrays are misaligned!"
