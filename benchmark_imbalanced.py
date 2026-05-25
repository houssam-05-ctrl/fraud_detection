import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import time

print("Loading dataset...")
# Load a subset for speed: all fraud cases, and a random sample of non-fraud cases
df = pd.read_csv('AIML Dataset.csv')
fraud_df = df[df['isFraud'] == 1]
non_fraud_df = df[df['isFraud'] == 0].sample(n=100000, random_state=42)
df_sampled = pd.concat([fraud_df, non_fraud_df]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset shape after sampling: {df_sampled.shape}")
print(f"Fraud cases: {len(fraud_df)}, Non-fraud cases: {len(non_fraud_df)}")

# Define features and target
categorical_features = ['type']
numerical_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

X = df_sampled[categorical_features + numerical_features]
y = df_sampled['isFraud']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

# Preprocess data
print("Preprocessing data...")
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

results = {}

# --- Method 1: SMOTE ---
print("\n--- Method 1: SMOTE ---")
start_time = time.time()
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_prep, y_train)

clf_smote = LogisticRegression(max_iter=1000)
clf_smote.fit(X_train_smote, y_train_smote)

preds_smote = clf_smote.predict(X_test_prep)
probs_smote = clf_smote.predict_proba(X_test_prep)[:, 1]

time_smote = time.time() - start_time
f1_smote = f1_score(y_test, preds_smote)
auc_smote = roc_auc_score(y_test, probs_smote)
print(f"SMOTE Training Time: {time_smote:.2f}s")
print("SMOTE Classification Report:")
print(classification_report(y_test, preds_smote))

results['SMOTE'] = {'F1-Score': f1_smote, 'AUC-ROC': auc_smote, 'Time(s)': time_smote}

# --- Method 2: Data Augmentation (Gaussian Noise) ---
print("\n--- Method 2: Data Augmentation (Gaussian Noise) ---")
start_time = time.time()

# Augment minority class by adding random Gaussian noise to numerical features
X_train_minority = X_train_prep[y_train == 1].toarray() if hasattr(X_train_prep, 'toarray') else X_train_prep[y_train == 1]
noise = np.random.normal(0, 0.1, X_train_minority.shape)
X_train_minority_aug = X_train_minority + noise

# Combine original and augmented
if hasattr(X_train_prep, 'toarray'):
    X_train_prep_dense = X_train_prep.toarray()
else:
    X_train_prep_dense = X_train_prep
    
X_train_aug = np.vstack([X_train_prep_dense, X_train_minority_aug])
y_train_aug = np.concatenate([y_train, np.ones(len(X_train_minority_aug))])

clf_aug = LogisticRegression(max_iter=1000)
clf_aug.fit(X_train_aug, y_train_aug)

preds_aug = clf_aug.predict(X_test_prep)
probs_aug = clf_aug.predict_proba(X_test_prep)[:, 1]

time_aug = time.time() - start_time
f1_aug = f1_score(y_test, preds_aug)
auc_aug = roc_auc_score(y_test, probs_aug)
print(f"Data Augmentation Training Time: {time_aug:.2f}s")
print("Data Augmentation Classification Report:")
print(classification_report(y_test, preds_aug))

results['Data Augmentation'] = {'F1-Score': f1_aug, 'AUC-ROC': auc_aug, 'Time(s)': time_aug}


# --- Method 3: Focal Loss (XGBoost) ---
print("\n--- Method 3: Focal Loss (XGBoost) ---")
start_time = time.time()

# Focal loss objective for xgboost
def focal_loss_xgb(preds, dtrain):
    labels = dtrain.get_label()
    alpha = 0.25
    gamma = 2.0
    p = 1.0 / (1.0 + np.exp(-preds))
    
    # Weight
    weight = np.where(labels == 1, alpha * (1 - p)**gamma, (1 - alpha) * p**gamma)
    
    # Gradients and Hessians 
    grad = np.where(labels == 1, p - 1.0, p) * weight
    hess = p * (1.0 - p) * weight
    return grad, hess

dtrain = xgb.DMatrix(X_train_prep, label=y_train)
dtest = xgb.DMatrix(X_test_prep, label=y_test)

params = {
    'learning_rate': 0.1,
    'max_depth': 6,
    'verbosity': 0
}

# Train using custom objective
gbm = xgb.train(params,
                dtrain,
                num_boost_round=100,
                obj=focal_loss_xgb)

probs_focal = gbm.predict(dtest)
probs_focal_sigmoid = 1.0 / (1.0 + np.exp(-probs_focal)) # since custom objective outputs logits
preds_focal = (probs_focal_sigmoid > 0.5).astype(int)

time_focal = time.time() - start_time
f1_focal = f1_score(y_test, preds_focal)
auc_focal = roc_auc_score(y_test, probs_focal_sigmoid)
print(f"Focal Loss Training Time: {time_focal:.2f}s")
print("Focal Loss Classification Report:")
print(classification_report(y_test, preds_focal))

results['Focal Loss'] = {'F1-Score': f1_focal, 'AUC-ROC': auc_focal, 'Time(s)': time_focal}


# --- Benchmark Summary ---
print("\n" + "="*40)
print(" BENCHMARK SUMMARY")
print("="*40)
summary_df = pd.DataFrame(results).T
print(summary_df)
summary_df.to_csv("benchmark_results.csv")
print("="*40)
print("Benchmark completed. Results saved to benchmark_results.csv")
