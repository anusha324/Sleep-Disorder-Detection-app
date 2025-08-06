import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# =============================================================================
# 1. DATA LOADING
# =============================================================================
print("=" * 60)
print("1. DATA LOADING")
print("=" * 60)

df = pd.read_csv('sleep_disorder_dataset.csv')

# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n" + "=" * 60)
print("2.PREPROCESSING AND EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

print(df.head(3))
print(df.shape)
print(df.columns)
print(df.isna().sum())

print(sum(df['Sleep Disorder'] == 'No disorder'))
df.fillna('No disorder',inplace = True)

print(df.duplicated().sum())

# ➤ Correlation Matrix (needs numerical columns only)
plt.figure(figsize=(10, 6))
corr = df.select_dtypes(include='number').corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# ➤ Sleep Disorder by BMI Category
plt.figure(figsize=(6, 4))
sns.countplot(x='BMI Category', hue='Sleep Disorder', data=df)
plt.title('Sleep Disorder by BMI Category')
plt.show()

#Sleep Disorder by Physical Activity Level
sleep_physical = df.groupby('Sleep Disorder')['Physical Activity Level'].value_counts()
sleep_physical = sleep_physical.reset_index()
plt.figure(figsize=(8, 5))
sns.lineplot(x = 'Physical Activity Level',
             y = 'count',
             hue = 'Sleep Disorder',
             data = sleep_physical,
             style = 'Sleep Disorder')
plt.xlabel('Physical Activity Level')
plt.ylabel('Count')
plt.title('Sleep Disorder by Physical Activity Level', loc='left')
plt.legend(title='Sleep Disorder Type')
plt.show()

#Sleep Disorder by Duration of Sleep
sleep_duration = df.groupby('Sleep Disorder')['Sleep Duration'].value_counts()
sleep_duration = sleep_duration.reset_index()
plt.figure(figsize=(8,5))
sns.lineplot(x = 'Sleep Duration',
             y = 'count',
             hue = 'Sleep Disorder',
             data = sleep_duration,
             style = 'Sleep Disorder')

plt.xlabel('Duration of Sleep')
plt.ylabel('Count')
plt.title('Sleep Disorder by Duration of Sleep', loc='left')
plt.legend(title='Sleep Disorder Type')
plt.show()

# ➤ Distribution of Sleep Disorder Classes
plt.figure(figsize=(6, 4))
sns.countplot(x='Sleep Disorder', data=df)
plt.title('Distribution of Sleep Disorder Types')
plt.show()

# ➤ Age Distribution by Sleep Disorder
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x='Age', hue='Sleep Disorder', kde=True, multiple='stack')
plt.title('Age Distribution by Sleep Disorder')
plt.show()

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 60)
print("3. FEATURE ENGINEERING")
print("=" * 60)

data = df.copy()
data = data.drop(['Person ID', 'Blood Pressure','Occupation'], axis=1)

# ➤ NEW FEATURE: Sleep Efficiency Score (Quality/Duration ratio)
data['Sleep_Efficiency'] = data['Quality of Sleep'] / data['Sleep Duration']
data['Sleep_Efficiency'] = data['Sleep_Efficiency'].clip(0.1, 2.0)  # Clip outliers

label_encoders = {}
cat_columns = ['BMI Category', 'Gender']
for col in cat_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
for col, le in label_encoders.items():
    joblib.dump(le, f'{col}_encoder.pkl')

# Manual mapping
disorder_mapping = {'No disorder': 0, 'Insomnia': 1, 'Sleep Apnea': 2}
# Apply the mapping to the column
data['Sleep Disorder'] = data['Sleep Disorder'].map(disorder_mapping)
# Save the mapping dictionary
joblib.dump(disorder_mapping, 'sleep_disorder_mapping_encoder.pkl')


print(data['Sleep Disorder'].value_counts())

"Feature correlation with target"
corrs = data.corr()['Sleep Disorder'].abs().sort_values(ascending=False)
print('Feature correlations with target:')
print(corrs)

print(data.columns)

# =============================================================================
# 4. DATA SPLITTING
# =============================================================================
print("\n" + "=" * 60)
print("4. DATA SPLITTING")
print("=" * 60)

X = data.drop(['Sleep Disorder'], axis=1) 
y = data['Sleep Disorder']

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0, stratify=y)

# =============================================================================
# 5. FEATURE SCALING
# =============================================================================
print("\n" + "=" * 60)
print("5. FEATURE SCALING")
print("=" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Save the scaler using joblib
joblib.dump(scaler, 'scaler.pkl')

# =============================================================================
# 6. MODEL SELECTION & CROSS-VALIDATION
# =============================================================================
print("\n" + "=" * 60)
print("6. MODEL SELECTION & CROSS-VALIDATION")
print("=" * 60)

# ➤ CROSS-VALIDATION
print("🔄 Cross-Validation Results:")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Test multiple models with cross-validation
models = {
    'XGBoost': XGBClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# =============================================================================
# 7. HYPERPARAMETER TUNING
# =============================================================================
print("\n" + "=" * 60)
print("7. HYPERPARAMETER TUNING")
print("=" * 60)

# ➤ HYPERPARAMETER TUNING for XGBoost
print("\n🔧 Hyperparameter Tuning...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

xgb_base = XGBClassifier(random_state=42, eval_metric='mlogloss')
random_search = RandomizedSearchCV(
    xgb_base, param_grid, n_iter=20, cv=3, scoring='accuracy', 
    random_state=42, n_jobs=-1
)
random_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.3f}")

# =============================================================================
# 8. MODEL TRAINING
# =============================================================================
print("\n" + "=" * 60)
print("8. MODEL TRAINING")
print("=" * 60)

# Use best model
xgb_model = random_search.best_estimator_

# Save model
joblib.dump(xgb_model, 'xgb_sleep_model.pkl')

# =============================================================================
# 9. MODEL EVALUATION
# =============================================================================
print("\n" + "=" * 60)
print("9. MODEL EVALUATION")
print("=" * 60)

# -----------------------------
# Enhanced Evaluation
# -----------------------------
y_pred = xgb_model.predict(X_test_scaled)
y_pred_proba = xgb_model.predict_proba(X_test_scaled)

print("🔍 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# ➤ ROC-AUC for each class
print("\n📈 ROC-AUC Scores:")
for i, class_name in enumerate(['No disorder', 'Insomnia', 'Sleep Apnea']):
    auc_score = roc_auc_score((y_test == i).astype(int), y_pred_proba[:, i])
    print(f"{class_name}: {auc_score:.3f}")

# ➤ Macro and Weighted AUC
macro_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
weighted_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
print(f"Macro AUC: {macro_auc:.3f}")
print(f"Weighted AUC: {weighted_auc:.3f}")

# =============================================================================
# 10. FEATURE IMPORTANCE ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("10. FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

# ➤ Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🎯 Top 5 Most Important Features:")
print(feature_importance.head())

# ➤ Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Top 10 Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

# =============================================================================
# 11. MODEL VISUALIZATION
# =============================================================================
print("\n" + "=" * 60)
print("11. MODEL VISUALIZATION")
print("=" * 60)

# ➤ ROC Curves for each class
plt.figure(figsize=(12, 4))
for i, class_name in enumerate(['No disorder', 'Insomnia', 'Sleep Apnea']):
    fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_pred_proba[:, i])
    auc_score = roc_auc_score((y_test == i).astype(int), y_pred_proba[:, i])
    plt.subplot(1, 3, i+1)
    plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {class_name}')
    plt.legend()
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 60)



