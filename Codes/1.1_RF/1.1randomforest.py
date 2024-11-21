import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Part 1: Synthetic Data Generation
# ---------------------------------
np.random.seed(42)
num_test_cases = 1000

test_case_id = [f"TC_{i+1}" for i in range(num_test_cases)]
execution_count = np.random.randint(1, 100, num_test_cases)
failure_rate = np.random.uniform(0, 1, num_test_cases)
code_coverage = np.random.uniform(0.5, 1.0, num_test_cases)
time_since_last_execution = np.random.randint(0, 365, num_test_cases)
test_case_priority = np.random.choice(['Low', 'Medium', 'High'], num_test_cases)
module_component = np.random.choice(['Module_A', 'Module_B', 'Module_C'], num_test_cases)
complexity = np.random.randint(1, 10, num_test_cases)

defect_scores = (0.3 * execution_count + 
                 0.5 * failure_rate + 
                 0.2 * code_coverage + 
                 np.random.normal(0, 0.1, num_test_cases))

thresholds = np.sort(np.random.choice(defect_scores, 4, replace=False))
defect_likelihood = np.digitize(defect_scores, bins=thresholds, right=True) + 1

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
data.to_csv('synthetic_test_data.csv', index=False)
print(data['defect_likelihood'].value_counts())
print(data.head())

# Part 2: Data Preprocessing
# --------------------------
data = pd.read_csv('synthetic_test_data.csv')
categorical_features = ['test_case_priority', 'module_component']
encoder = OneHotEncoder(drop='first', sparse_output=False)

encoded_features = pd.DataFrame(encoder.fit_transform(data[categorical_features]), 
                                columns=encoder.get_feature_names_out(categorical_features))

data_preprocessed = pd.concat([data.drop(columns=categorical_features), encoded_features], axis=1)
data_preprocessed.to_csv('preprocessed_test_data.csv', index=False)
print(data_preprocessed.head())

# Part 3: Model Training with Random Forest
# -----------------------------------------
data_preprocessed = pd.read_csv('preprocessed_test_data.csv')

features = data_preprocessed.drop(columns=['test_case_id', 'defect_likelihood'])
target = data_preprocessed['defect_likelihood']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("Model Performance:")
print(classification_report(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))

feature_importances = pd.DataFrame(rf.feature_importances_, index=features.columns, columns=['importance']).sort_values(by='importance', ascending=False)
print("Feature Importances:")
print(feature_importances)
feature_importances.to_csv('feature_importances.csv')

# Part 4: Test Case Ranking and Visualization
# -------------------------------------------
defect_probabilities = rf.predict_proba(features)

data_preprocessed['defect_probability'] = defect_probabilities.max(axis=1)
data_preprocessed['predicted_class'] = rf.predict(features)

ranked_test_cases = data_preprocessed[['test_case_id', 'execution_count', 'failure_rate', 'defect_probability', 'predicted_class']].sort_values(by='defect_probability', ascending=False)

print("Top Test Cases by Defect Probability:")
print(ranked_test_cases.head())
ranked_test_cases.to_csv('ranked_test_cases.csv', index=False)

plt.figure(figsize=(12, 8))
for class_value in sorted(ranked_test_cases['predicted_class'].unique()):
    class_data = ranked_test_cases[ranked_test_cases['predicted_class'] == class_value]
    plt.scatter(class_data['execution_count'], class_data['failure_rate'], label=f'Class {class_value}', alpha=0.6)

plt.xlabel('Execution Count')
plt.ylabel('Failure Rate')
plt.title('Execution Count vs. Failure Rate by Predicted Class')
plt.legend(title='Predicted Class')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 8))
ranked_test_cases.boxplot(column='defect_probability', by='predicted_class', grid=False, patch_artist=True)
plt.xlabel('Predicted Class')
plt.ylabel('Defect Probability')
plt.title('Box Plot of Defect Probability by Predicted Class')
plt.suptitle('')
plt.show()

# Part 5: Model Evaluation Visualization
# --------------------------------------
# Confusion Matrix Plot
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Random Forest')
plt.show()
