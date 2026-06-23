import os
import joblib
import numpy as np
import sys
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier

# ==============================================================================
# 1. THE CLASS BLUEPRINT (Required for joblib to successfully load your model)
# ==============================================================================
class WeakMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_sizes=(32,), max_iter=15, activation='relu',
                 alpha=1e-3, batch_size=64, early_stopping=True, validation_fraction=0.1,
                 n_iter_no_change=5, use_class_balance=False, random_state=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.activation = activation
        self.alpha = alpha
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.use_class_balance = use_class_balance
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None): pass
    def predict(self, X): return self.mlp_.predict(X)
    def predict_proba(self, X): return self.mlp_.predict_proba(X)

# ==============================================================================
# 2. INFERENCE PIPELINE
# ==============================================================================
def predict(features_array):
    """
    Args:
        features_array (np.ndarray): The 1024-dimensional extracted features.
    """
    # 1. Locate the disguised model inside the checkpoints folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "checkpoints", "best_model.pt")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model missing at {model_path}")

    # 2. Load the AdaBoost model
    model = joblib.load(model_path)
    
    # 3. Predict (Outputs 1, 2, 3, 4)
    raw_predictions = model.predict(features_array)
    
    # 4. Map 1, 2, 3, 4 to 'A', 'B', 'C', 'D'
    label_map = np.array(['A', 'B', 'C', 'D'])
    indices = raw_predictions.astype(int) - 1  # Shift to 0, 1, 2, 3 for array indexing
    
    return label_map[indices]

# ==============================================================================
# 3. COMMAND LINE EXECUTION (Used by the benchmark bash script)
# ==============================================================================
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict.py <input.npy> <output.npy>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    X_test = np.load(input_path)
    final_preds = predict(X_test)
    np.save(output_path, final_preds)
    print(f"Predictions successfully mapped to A/B/C/D and saved to {output_path}")
