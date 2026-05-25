import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb

print("Chargement des données...")
df = pd.read_csv('AIML Dataset.csv')
fraud_df = df[df['isFraud'] == 1]
non_fraud_df = df[df['isFraud'] == 0].sample(n=100000, random_state=42)
df_sampled = pd.concat([fraud_df, non_fraud_df]).sample(frac=1, random_state=42).reset_index(drop=True)

categorical_features = ['type']
numerical_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

X = df_sampled[categorical_features + numerical_features]
y = df_sampled['isFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

# Focal Loss
def focal_loss_xgb(preds, dtrain):
    labels = dtrain.get_label()
    alpha = 0.25
    gamma = 2.0
    p = 1.0 / (1.0 + np.exp(-preds))
    weight = np.where(labels == 1, alpha * (1 - p)**gamma, (1 - alpha) * p**gamma)
    grad = np.where(labels == 1, p - 1.0, p) * weight
    hess = p * (1.0 - p) * weight
    return grad, hess

dtrain = xgb.DMatrix(X_train_prep, label=y_train)
dtest = xgb.DMatrix(X_test_prep, label=y_test)

evals_result = {}
params = {
    'learning_rate': 0.1,
    'max_depth': 6,
    'eval_metric': 'logloss',
    'verbosity': 0
}

print("Entraînement de XGBoost et enregistrement des métriques...")
gbm = xgb.train(params,
                dtrain,
                num_boost_round=150,
                evals=[(dtrain, 'train'), (dtest, 'val')],
                obj=focal_loss_xgb,
                evals_result=evals_result,
                verbose_eval=False)

# Tracé
print("Génération du graphique...")
epochs = len(evals_result['train']['logloss'])
x_axis = range(0, epochs)

# Design épuré moderne style sombre
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#121212')
ax.set_facecolor('#1e1e1e')

ax.plot(x_axis, evals_result['train']['logloss'], label='Perte Entraînement (Train Loss)', color='#00f2fe', linewidth=2.5)
ax.plot(x_axis, evals_result['val']['logloss'], label='Perte Validation (Val Loss)', color='#f35588', linewidth=2.5)

ax.grid(True, color='#444444', linestyle='--', alpha=0.5)
ax.set_title("Courbes d'Apprentissage (Diagnostic d'Overfitting / Underfitting)", fontsize=14, fontweight='bold', pad=15, color='#ffffff')
ax.set_xlabel('Itérations (Boosting Rounds)', fontsize=11, color='#cccccc')
ax.set_ylabel('Log Loss (BCE)', fontsize=11, color='#cccccc')
ax.tick_params(colors='#cccccc')

# Annotation pour expliquer la zone
ax.legend(facecolor='#1e1e1e', edgecolor='#444444', fontsize=10)

plt.tight_layout()
plt.savefig('learning_curves.png', dpi=300, facecolor='#121212')
print("Graphique sauvegardé avec succès sous 'learning_curves.png'")
