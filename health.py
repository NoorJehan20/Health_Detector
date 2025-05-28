import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("heart_cleveland_upload.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.shape)
print(df.isnull().sum())

df.rename(columns={
    'cp': 'chest_pain_type',
    'trestbps': 'resting_bp',
    'chol': 'cholesterol'
}, inplace=True)
print(df.head())

#Histograms + KDE overlays for numeric columns
numeric_columns = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(16, 12))

for i, col in enumerate(numeric_columns, 1):
    plt.subplot(4, 4, i)
    sns.histplot(df[col], bins=20, kde=True, color='teal', stat='density', edgecolor='black', alpha=0.6)
    plt.title(f'Distribution & KDE of {col}')
    plt.xlabel(col)
    plt.ylabel('Density')

plt.tight_layout()
plt.suptitle('Histograms & KDE Plots for Numeric Features', y=1.02)
plt.show()

#Correlation Heatmap with Clustering
numeric_df = df[numeric_columns]
if numeric_df.shape[1] >= 4:
    sns.clustermap(numeric_df.corr(), annot=True, cmap='coolwarm', figsize=(12, 10), fmt='.2f', linewidths=0.5)
    plt.title('Clustered Correlation Heatmap', pad=120)
    plt.show()

#Countplot with percentage annotations for 'condition'
plt.figure(figsize=(8,5))
ax = sns.countplot(x='condition', data=df, palette='viridis')
total = len(df)
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height} ({height/total:.1%})', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=12)
plt.title('Countplot of Condition with Percentages')
plt.xlabel('Condition')
plt.ylabel('Count')
plt.show()

#Features for violin and KDE plots
important_feature_violin = 'cholesterol'
important_feature_kde = 'age'

#Violin plot for 'cholesterol' by target
plt.figure(figsize=(8,4))
sns.violinplot(x='condition', y=important_feature_violin, data=df, palette='viridis')
plt.title(f'Violin Plot of {important_feature_violin} by Condition')
plt.show()

#FacetGrid KDE plot for 'age' by target
g = sns.FacetGrid(df, hue='condition', height=4, aspect=1.5, palette='viridis')
g.map(sns.kdeplot, important_feature_kde, fill=True, alpha=0.4)
g.add_legend(title='Condition')
plt.title(f'KDE Plot of {important_feature_kde} by Condition')
plt.show()

# Preparing features and target function
def prepare_features_target(df, target_col='condition', categorical_cols=None, drop_first=True):
    """
    Prepare feature matrix X and target vector y for modeling.
    """
    df_model = df.copy()

#Detecting categorical columns
    if categorical_cols is None:
        categorical_cols = df_model.select_dtypes(include=['object', 'category']).columns.tolist()

#Applying one-hot encoding
    if categorical_cols:
        df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=drop_first)

#Separating target and features
    if target_col in df_model.columns:
        y = df_model[target_col].astype('int')
        X = df_model.drop(columns=[target_col])
    else:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    return X, y

#Spliting Data
from sklearn.model_selection import train_test_split

X, y = prepare_features_target(df, target_col='condition')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Seting_up models and GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

#Defining pipelines and hyperparameters
logreg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])
logreg_params = {
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__penalty': ['l2'],
    'clf__solver': ['lbfgs']
}

rf_pipeline = Pipeline([
    ('clf', RandomForestClassifier(random_state=42))
])
rf_params = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 10, 20],
    'clf__min_samples_split': [2, 5]
}

svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(probability=True))
])
svm_params = {
    'clf__C': [0.1, 1, 10],
    'clf__kernel': ['linear', 'rbf']
}

#Running GridSearchCV for each model
grids = {
    'Logistic Regression': (logreg_pipeline, logreg_params),
    'Random Forest': (rf_pipeline, rf_params),
    'SVM': (svm_pipeline, svm_params)
}

best_models = {}

for name, (pipe, params) in grids.items():
    print(f"\nRunning GridSearchCV for: {name}")
    grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_models[name] = grid.best_estimator_
    print("Best Params:", grid.best_params_)

    y_pred = grid.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

best_models['Logistic Regression']

#Confusion Matrix for models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

for name, model in best_models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    print(f"\nConfusion Matrix for {name}:")
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.show()

#ROC Curves
from sklearn.metrics import roc_curve, roc_auc_score

plt.figure(figsize=(8, 6))

for name, model in best_models.items():
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)[:, 1]
    else:
#For models like SVM without predict_proba enabled
        y_probs = model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    auc_score = roc_auc_score(y_test, y_probs)

    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid()
plt.show()

import joblib
joblib.dump(best_models['Random Forest'], 'best_model.pkl')