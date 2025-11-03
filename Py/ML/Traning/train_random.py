import os, sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from RandomForest import RandomForest



current_dir = os.path.dirname(__file__)
root_dir = os.path.dirname(current_dir)

EXTRACTED_DATA_PATH = os.path.join(current_dir, "features_dataset.xlsx")
MODEL_PATH = os.path.join(root_dir, "rf_model.pkl")



df = pd.read_excel(EXTRACTED_DATA_PATH)
print(f"[INFO] Loaded dataset from: {EXTRACTED_DATA_PATH}")

X = df.drop(columns=["Label"]).values
y = df["Label"].values

print(f"[INFO] Dataset shape: {X.shape}, Labels: {np.unique(y, return_counts=True)}")



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234, stratify=y
)



def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)



print("[INFO] Training Random Forest model...")
clf = RandomForest(
    n_trees=100,        
    max_depth=15,       
    min_samples_split=2
)
clf.fit(X_train, y_train)


predictions = clf.predict(X_test)
acc = accuracy(y_test, predictions)
print(f"[INFO] RandomForest Accuracy: {acc:.4f}")


cm = confusion_matrix(y_test, predictions)
print("[INFO] Confusion Matrix:")
print(cm)



print("\n[INFO] Running 5-Fold Cross-Validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=1234)
fold_acc = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    
    model = RandomForest(n_trees=100, max_depth=15)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    fold_acc.append(accuracy(y_te, pred))
    print(f"  Fold {fold} Accuracy: {fold_acc[-1]:.4f}")

print(f"[INFO] Average 5-Fold Accuracy: {np.mean(fold_acc):.4f}")


if os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)
    print(f"[INFO] Existing model deleted: {MODEL_PATH}")

with open(MODEL_PATH, "wb") as f:
    pickle.dump(clf, f)

print(f"[INFO] New Random Forest model saved as {MODEL_PATH}")
