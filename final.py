"""
TP3 - Pneumonia detection from X-ray images
Simplified version using train + test folders
Feature extraction methods: 
A - First-order statistics
B - GLCM
C - Wavelet
D - HOG
E - LBP
"""

import os
import random
import warnings
import numpy as np
from glob import glob
from PIL import Image
from scipy import stats
import pywt
from skimage.feature import hog, local_binary_pattern
from skimage.feature.texture import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")

# -------------------- CONFIG --------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

IMG_SIZE = (128, 128)

TRAIN_DIR = r"/home/farouk/Downloads/TP3 download/database_Pneumonia300/train"
TEST_DIR  = r"/home/farouk/Downloads/TP3 download/database_Pneumonia300/test"

NORMAL = "Normal"
PNEUMONIA = "Pneumonia"

NUM_TRAIN = 100   # images per class in training
NUM_TEST  = 50    # images per class in testing
# -------------------------------------------------

def load_gray_image(path):
    """Load an image, convert to grayscale, and resize."""
    try:
        img = Image.open(path).convert("L").resize(IMG_SIZE)
        return np.array(img)
    except:
        print(f"[WARN] Could not load {path}. Returning blank image.")
        return np.zeros(IMG_SIZE, dtype=np.uint8)

def get_random_files(folder, num):
    """Return a random subset of image paths from a folder."""
    all_files = glob(os.path.join(folder, "*.*"))
    if len(all_files) == 0:
        print(f"[WARN] No images found in {folder}")
    random.shuffle(all_files)
    return all_files[:num]

def prepare_dataset(train_root, test_root):
    """Build a list of samples: (filepath, label, set_type)"""
    samples = []
    for cls in (NORMAL, PNEUMONIA):
        label = 0 if cls == NORMAL else 1

        # Training samples
        train_files = get_random_files(os.path.join(train_root, cls), NUM_TRAIN)
        for f in train_files:
            samples.append((f, label, "train"))
        print(f"[INFO] Loaded {len(train_files)} training images for {cls}")

        # Testing samples
        test_files = get_random_files(os.path.join(test_root, cls), NUM_TEST)
        for f in test_files:
            samples.append((f, label, "val"))
        print(f"[INFO] Loaded {len(test_files)} testing images for {cls}")

    return samples

# ---------------- Feature Methods ----------------
def first_order_features(img):
    """Compute basic statistics from image pixel values."""
    pixels = img.ravel().astype(np.float32)
    return np.array([
        pixels.mean(), pixels.std(), np.median(pixels),
        pixels.min(), pixels.max(),
        float(stats.skew(pixels)),
        float(stats.kurtosis(pixels)),
        float(shannon_entropy(img, base=2))
    ], dtype=np.float32)

def glcm_features(img):
    """Compute GLCM features: contrast, homogeneity, energy, correlation."""
    img_q = (img * 31 / 255).astype(np.uint8)
    glcm = graycomatrix(img_q, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=32, symmetric=True, normed=True)
    feats = []
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        feats.extend(graycoprops(glcm, prop).ravel())
    return np.array(feats, dtype=np.float32)

def wavelet_features(img):
    """Compute wavelet transform statistics."""
    coeffs = pywt.wavedec2(img, 'db1', level=2)
    feats = []
    for c in coeffs:
        if isinstance(c, tuple):
            c = np.hstack(c)
        feats += [np.mean(c), np.std(c)]
    return np.array(feats, dtype=np.float32)

def hog_features(img):
    """Compute HOG features."""
    return hog(img, orientations=9, pixels_per_cell=(16,16),
               cells_per_block=(2,2), feature_vector=True).astype(np.float32)

def lbp_features(img):
    """Compute LBP histogram."""
    lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=int(lbp.max()+1), density=True)
    return hist.astype(np.float32)

# --------------- Feature Matrix Builder ---------------
def build_feature_matrix(samples, feat_func):
    X, y, sets = [], [], []
    for path, label, set_type in samples:
        feats = feat_func(load_gray_image(path))
        if feats is not None and len(feats) > 0:
            X.append(feats)
            y.append(label)
            sets.append(set_type)
    if len(X) == 0:
        raise ValueError("[ERROR] No features extracted! Check paths.")
    return np.vstack(X), np.array(y), np.array(sets)

# ---------------- Train & Evaluate ----------------
def train_and_evaluate(X, y, sets):
    """Train an MLP classifier and return metrics."""
    X_train, X_val = X[sets=="train"], X[sets=="val"]
    y_train, y_val = y[sets=="train"], y[sets=="val"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=RANDOM_SEED)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_val_scaled)
    y_prob = clf.predict_proba(X_val_scaled)[:,1]

    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "auc": roc_auc_score(y_val, y_prob)
    }

# ---------------- Main Script ----------------
if __name__ == "__main__":
    print("\nLoading dataset...")
    samples = prepare_dataset(TRAIN_DIR, TEST_DIR)
    print(f"Total images loaded: {len(samples)}\n")

    feature_methods = {
        "A_FirstOrder": first_order_features,
        "B_GLCM": glcm_features,
        "C_Wavelet": wavelet_features,
        "D_HOG": hog_features,
        "E_LBP": lbp_features
    }

    # Print metrics table
    print("="*70)
    print("{:<15} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Method", "Accuracy", "Precision", "Recall", "F1", "AUC"
    ))
    print("="*70)

    for name, func in feature_methods.items():
        X, y, sets = build_feature_matrix(samples, func)
        metrics = train_and_evaluate(X, y, sets)

        print("{:<15} {:>9.2f}% {:>9.2f}% {:>9.2f}% {:>9.2f}% {:>9.2f}%".format(
            name,
            metrics["accuracy"]*100,
            metrics["precision"]*100,
            metrics["recall"]*100,
            metrics["f1"]*100,
            metrics["auc"]*100
        ))

    print("="*70)
    print("\nDone!")
