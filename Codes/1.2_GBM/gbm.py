import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import lightgbm as lgb
import xgboost as xgb

# --- Step 1: Create Enhanced Synthetic Dataset ---
np.random.seed(42)
num_test_cases = 2000  # Increased number for better training

# Generate synthetic fields with complex patterns
test_case_id = [f"TC_{i+1}" for i in range(num_test_cases)]
execution_count = np.random.randint(1, 200, num_test_cases)
failure_rate = np.random.uniform(0, 1, num_test_cases)
code_coverage = np.random.uniform(0.4, 1.0, num_test_cases)  # Adjusted range
time_since_last_execution = np.random.randint(0, 730, num_test_cases)  # Increased range
test_case_priority = np.random.choice(['Low', 'Medium', 'High'], num_test_cases)
module_component = np.random.choice(['Module_A', 'Module_B', 'Module_C', 'Module_D'], num_test_cases)
complexity = np.random.randint(1, 20, num_test_cases)  # Increased range

# Generate defect scores with more complex patterns
defect_scores = (0.25 * execution_count + 
                 0.35 * failure_rate + 
                 0.25 * code_coverage + 
                 0.1 * np.sin(time_since_last_execution / 100) + 
                 0.05 * np.log1p(complexity) + 
                 np.random.normal(0, 0.3, num_test_cases))  # Increased noise

# Create defect likelihood with more granularity
bins = np.percentile(defect_scores, [10, 25, 50, 75, 90])
defect_likelihood = np.digitize(defect_scores, bins=bins, right=True)
defect_likelihood = np.clip(defect_likelihood, 1, 5)  # Ensure values are within range

# Shuffle the dataset
data = pd.DataFrame({
    'test_case_id': test_case_id,
    'execution_count': execution_count,
    'failure_rate': failure_rate,
    'code_coverage': code_coverage,
    'time_since_last_execution': time_since_last_execution,
    'test_case_priority': test_case_priority,
    'module_component': module_component,
    'complexity': complexity,
    'defect_likelihood': defect_likelihood
})
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the synthetic dataset to a CSV file
data.to_csv('synthetic_test_data.csv', index=False)
print(data['defect_likelihood'].value_counts())
print(data.head())

# --- Step 2: Preprocess the Dataset ---
data_preprocessed = pd.read_csv('synthetic_test_data.csv')

# Encode categorical variables
categorical_features = ['module_component', 'test_case_priority']
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_features = pd.DataFrame(encoder.fit_transform(data_preprocessed[categorical_features]), 
                                columns=encoder.get_feature_names_out(categorical_features))

# Combine encoded features with the original data
data_preprocessed = pd.concat([data_preprocessed.drop(columns=categorical_features), encoded_features], axis=1)

# Normalize numerical features
scaler = StandardScaler()
numerical_features = ['execution_count', 'failure_rate', 'code_coverage', 'time_since_last_execution', 'complexity']
data_preprocessed[numerical_features] = scaler.fit_transform(data_preprocessed[numerical_features])

# Feature selection
selector = SelectKBest(score_func=f_classif, k='all')
data_preprocessed[numerical_features] = selector.fit_transform(data_preprocessed[numerical_features], data_preprocessed['defect_likelihood'])

# Adjust target labels to start from 0
data_preprocessed['defect_likelihood'] -= 1

# Save the preprocessed data to a new file
data_preprocessed.to_csv('preprocessed_test_data.csv', index=False)

# --- Step 3: Train the Neural Network Model ---
features = data_preprocessed.drop(columns=['test_case_id', 'defect_likelihood'])
target = data_preprocessed['defect_likelihood']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define Neural Network model
model = Sequential()
model.add(Dense(128, input_dim=X_train_resampled.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))  # Adjust output layer for multi-class

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model
history = model.fit(X_train_resampled, y_train_resampled, epochs=50, batch_size=32, validation_split=0.2, verbose=2)

# Evaluate the model
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

print("Neural Network Model Performance:")
print(classification_report(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))

# Save Neural Network predictions
nn_predictions = pd.DataFrame({
    'test_case_id': data_preprocessed.loc[X_test.index, 'test_case_id'],
    'true_label': y_test,
    'predicted_label': y_pred
})
nn_predictions.to_csv('nn_predictions.csv', index=False)

# --- Visualization: Training History ---
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Neural Network Training History')
plt.legend()
plt.show()

# --- Visualization: Confusion Matrix for Neural Network ---
conf_matrix_nn = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_nn, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Neural Network')
plt.show()

# --- Step 4: Train the LightGBM Model ---
lgbm = lgb.LGBMClassifier(random_state=42)
lgbm.fit(X_train_resampled, y_train_resampled)

# Evaluate LightGBM
y_pred_lgbm = lgbm.predict(X_test)
print("LightGBM Performance:")
print(classification_report(y_test, y_pred_lgbm))
print('Accuracy:', accuracy_score(y_test, y_pred_lgbm))

# Save LightGBM predictions
lgbm_predictions = pd.DataFrame({
    'test_case_id': data_preprocessed.loc[X_test.index, 'test_case_id'],
    'true_label': y_test,
    'predicted_label': y_pred_lgbm
})
lgbm_predictions.to_csv('lgbm_predictions.csv', index=False)

# Get feature importances from LightGBM
lgbm_feature_importances = pd.DataFrame({'Feature': features.columns, 'Importance': lgbm.feature_importances_})
lgbm_feature_importances = lgbm_feature_importances.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=lgbm_feature_importances, palette='viridis')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance - LightGBM')
plt.show()

# Save LightGBM feature importances
lgbm_feature_importances.to_csv('lgbm_feature_importances.csv', index=False)

# --- Visualization: Confusion Matrix for LightGBM ---
conf_matrix_lgbm = confusion_matrix(y_test, y_pred_lgbm)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_lgbm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - LightGBM')
plt.show()

# --- Step 5: Train the XGBoost Model ---
xgbm = xgb.XGBClassifier(random_state=42)
xgbm.fit(X_train_resampled, y_train_resampled)

# Evaluate XGBoost
y_pred_xgbm = xgbm.predict(X_test)
print("XGBoost Performance:")
print(classification_report(y_test, y_pred_xgbm))
print('Accuracy:', accuracy_score(y_test, y_pred_xgbm))

# Save XGBoost predictions
xgbm_predictions = pd.DataFrame({
    'test_case_id': data_preprocessed.loc[X_test.index, 'test_case_id'],
    'true_label': y_test,
    'predicted_label': y_pred_xgbm
})
xgbm_predictions.to_csv('xgbm_predictions.csv', index=False)

# Save XGBoost feature importances
xgbm_feature_importances = pd.DataFrame({'Feature': features.columns, 'Importance': xgbm.feature_importances_})
xgbm_feature_importances = xgbm_feature_importances.sort_values(by='Importance', ascending=False)
xgbm_feature_importances.to_csv('xgbm_feature_importances.csv', index=False)

# --- Visualization: Confusion Matrix for XGBoost ---
conf_matrix_xgbm = confusion_matrix(y_test, y_pred_xgbm)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_xgbm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - XGBoost')
plt.show()

# --- Visualization: Distribution of Predicted Probabilities ---
plt.figure(figsize=(10, 6))
sns.histplot(y_pred_prob.max(axis=1), bins=10, kde=True)
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Distribution of Maximum Predicted Probabilities')
plt.show()
