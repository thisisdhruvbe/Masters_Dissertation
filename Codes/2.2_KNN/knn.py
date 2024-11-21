import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns

# Define the number of samples
num_samples = 200

# Generate synthetic data
np.random.seed(42)  # Ensure reproducibility

# Generate features with correlations
feature_1 = np.random.uniform(1, 100, num_samples)
feature_2 = 0.5 * feature_1 + np.random.normal(0, 10, num_samples)  # Correlated with feature_1
feature_3 = np.random.uniform(1, 50, num_samples)
feature_4 = 0.3 * feature_2 + np.random.normal(0, 5, num_samples)  # Correlated with feature_2

# Introduce more complex relationships
execution_time = 0.2 * feature_1 + 0.5 * feature_3 + np.random.normal(0, 10, num_samples)
complexity_index = np.sin(feature_1 / 10) + np.cos(feature_2 / 10) + np.random.normal(0, 0.5, num_samples)

# Generate target variable with some noise
def execute_outcome(feature_1, feature_2, feature_3, feature_4):
    # More complex rule-based outcome
    outcome = (feature_1 > 50) & (feature_2 > 30) | (feature_3 < 20) | (feature_4 < 0.5)
    return outcome.astype(int)

# Generate target variable
execution_outcome = execute_outcome(feature_1, feature_2, feature_3, feature_4)

data = {
    'Test Case ID': np.arange(1, num_samples + 1),
    'Execution Outcome': execution_outcome,
    'Feature_1': feature_1,
    'Feature_2': feature_2,
    'Feature_3': feature_3,
    'Feature_4': feature_4,
    'Execution Time (sec)': execution_time,
    'Complexity Index': complexity_index
}

df = pd.DataFrame(data)

# Save synthetic dataset
df.to_csv('synthetic_data_knn.csv', index=False)

# Define features and target variable
X = df[['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Execution Time (sec)', 'Complexity Index']]
y = df['Execution Outcome']

# Feature Selection
selector = SelectKBest(f_classif, k=5)  # Use ANOVA F-value for feature selection
X_selected = selector.fit_transform(X, y)

# Save selected features
selected_features_df = pd.DataFrame(X_selected, columns=['Feature_1', 'Feature_2', 'Feature_3', 'Execution Time (sec)', 'Complexity Index'])
selected_features_df.to_csv('selected_features_knn.csv', index=False)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_selected, y)

# Save resampled data
resampled_df = pd.DataFrame(X_resampled, columns=selected_features_df.columns)
resampled_df['Execution Outcome'] = y_resampled
resampled_df.to_csv('resampled_data_knn.csv', index=False)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaled data
scaled_train_df = pd.DataFrame(X_train, columns=selected_features_df.columns)
scaled_test_df = pd.DataFrame(X_test, columns=selected_features_df.columns)
scaled_train_df['Execution Outcome'] = y_train
scaled_test_df['Execution Outcome'] = y_test
scaled_train_df.to_csv('scaled_train_data_knn.csv', index=False)
scaled_test_df.to_csv('scaled_test_data_knn.csv', index=False)

# Define parameter grid for GridSearchCV
param_grid_knn = {
    'n_neighbors': [5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Initialize Grid Search with StratifiedKFold
grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='accuracy', n_jobs=-1)
grid_search_knn.fit(X_train, y_train)

# Best model from Grid Search
best_knn_model = grid_search_knn.best_estimator_

# Make predictions
y_pred_knn = best_knn_model.predict(X_test)
y_prob_knn = best_knn_model.predict_proba(X_test)[:, 1]

# Save predictions
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_knn, 'Probability': y_prob_knn})
predictions_df.to_csv('knn_predictions.csv', index=False)

# Model Evaluation
accuracy_knn = accuracy_score(y_test, y_pred_knn)
report_knn = classification_report(y_test, y_pred_knn)

# Print Accuracy and Classification Report
print("KNN Model:")
print(f"Accuracy: {accuracy_knn:.2f}")
print("Classification Report:")
print(report_knn)

# Save classification report
with open('knn_classification_report.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy_knn:.2f}\n")
    f.write("Classification Report:\n")
    f.write(report_knn)

# Confusion Matrix for KNN
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='Blues', xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - KNN')
plt.show()

# Save confusion matrix as CSV
conf_matrix_df = pd.DataFrame(conf_matrix_knn, index=['Fail', 'Pass'], columns=['Fail', 'Pass'])
conf_matrix_df.to_csv('knn_confusion_matrix.csv')

# Normalized Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_knn / np.sum(conf_matrix_knn), annot=True, fmt='.2%', cmap='Blues', xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Normalized Confusion Matrix - KNN')
plt.show()

# ROC Curve for KNN
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn, pos_label=1)
roc_auc_knn = auc(fpr_knn, tpr_knn)
plt.figure(figsize=(10, 7))
plt.plot(fpr_knn, tpr_knn, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_knn:.2f}) - KNN')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - KNN')
plt.legend(loc='lower right')
plt.show()

# Precision-Recall Curve for KNN
precision_knn, recall_knn, _ = precision_recall_curve(y_test, y_prob_knn, pos_label=1)
plt.figure(figsize=(10, 7))
plt.plot(recall_knn, precision_knn, color='blue', lw=2, label='KNN')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - KNN')
plt.show()

# Learning Curves for KNN
train_sizes, train_scores, test_scores = learning_curve(best_knn_model, X_resampled, y_resampled, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
plt.figure(figsize=(10, 7))
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='r', label='Training score - KNN')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color='g', label='Cross-validation score - KNN')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.title('Learning Curves - KNN')
plt.legend(loc='best')
plt.grid(True)
plt.show()
