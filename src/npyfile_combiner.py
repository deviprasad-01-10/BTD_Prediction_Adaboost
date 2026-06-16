import numpy as np

X_old = np.load(r"C:\...\train_X.npy")
y_old = np.load(r"C:\...\train_y.npy")
X_new = np.load(r"C:\...\train_X_2.npy")
y_new = np.load(r"C:\...\train_y_2.npy")

X_combined = np.vstack([X_old, X_new])
y_combined = np.concatenate([y_old, y_new])

np.save(r"C:\...\train_X_combined.npy", X_combined)
np.save(r"C:\...\train_y_combined.npy", y_combined)

print(f"Old: {X_old.shape} | New: {X_new.shape} | Combined: {X_combined.shape}")