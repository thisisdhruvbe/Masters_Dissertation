import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# 1. Enhanced Realistic Synthetic Data Creation with More Classifications
def create_enhanced_realistic_synthetic_data(num_samples=2000):
    categories = [
        'performance', 'security', 'usability', 'compatibility', 'reliability',
        'scalability', 'maintainability', 'portability', 'usability', 'interoperability'
    ]
    texts = []
    labels = []

    # Define class labels for the expanded categories
    category_labels = {
        'performance': 0, 'security': 1, 'usability': 2, 'compatibility': 3, 
        'reliability': 4, 'scalability': 5, 'maintainability': 6, 
        'portability': 7, 'interoperability': 8
    }

    for i in range(num_samples):
        category = np.random.choice(categories)
        text = f"Requirement related to {category} issues, consider reviewing the system's {category} aspects closely." if np.random.rand() > 0.5 else \
               f"Ensure the {category} components meet the latest standards. Potential issues might arise in the {category} domain."
        label = category_labels.get(category, 0)  # Default to 0 if not found
        texts.append(text)
        labels.append(label)
    
    df = pd.DataFrame({'Text': texts, 'Label': labels})
    
    # Save synthetic data to CSV
    df.to_csv('synthetic_data.csv', index=False)
    
    return df

# Create synthetic data
synthetic_data = create_enhanced_realistic_synthetic_data()

# 2. Data Preprocessing with Data Augmentation
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),  # Remove batch dimension
            'attention_mask': encoding['attention_mask'].squeeze(),  # Remove batch dimension
            'label': torch.tensor(label, dtype=torch.long)
        }

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(synthetic_data['Text'], synthetic_data['Label'], test_size=0.2, random_state=42)

# Create Datasets and DataLoaders
train_dataset = TextDataset(X_train.values, y_train.values, tokenizer)
val_dataset = TextDataset(X_val.values, y_val.values, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 3. Model Training and Fine-Tuning with Adjustments
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=9)  # Updated number of labels
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

def train_and_evaluate_model(num_epochs=10, patience=3):
    def scheduler_function(epoch):
        return max(1.0 - epoch / num_epochs, 0.1)

    # Use LambdaLR directly with a lambda function
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: scheduler_function(epoch))

    train_accuracies = []
    val_accuracies = []
    best_val_accuracy = 0
    epochs_without_improvement = 0

    all_val_labels = []
    all_val_preds = []

    for epoch in range(num_epochs):
        model.train()
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Calculate training accuracy
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

        train_accuracy = train_correct / train_total
        train_accuracies.append(train_accuracy)

        # Evaluation step
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(predictions.cpu().numpy())

        val_accuracy = val_correct / val_total
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}, Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Early stopping logic
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping due to no improvement.")
                break

    # Save training and validation accuracies to CSV
    results_df = pd.DataFrame({
        'Epoch': list(range(1, len(train_accuracies) + 1)),
        'Training Accuracy': train_accuracies,
        'Validation Accuracy': val_accuracies
    })
    results_df.to_csv('training_validation_accuracies.csv', index=False)
    
    # Plot confusion matrix
    plot_confusion_matrix(all_val_labels, all_val_preds)

def plot_confusion_matrix(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'Class {i}' for i in range(conf_matrix.shape[1])], 
                yticklabels=[f'Class {i}' for i in range(conf_matrix.shape[0])])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

# Predictive function for new examples
def predict_new_examples(model, tokenizer, examples):
    model.eval()
    predictions = []

    with torch.no_grad():
        for text in examples:
            encoding = tokenizer(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1).item()
            predictions.append(prediction)
    
    return predictions

# Save predictions to CSV
def save_predictions(examples, predicted_labels):
    predictions_df = pd.DataFrame({
        'Text': examples,
        'Predicted Label': predicted_labels
    })
    predictions_df.to_csv('predictions.csv', index=False)

# Run the training and evaluation
train_and_evaluate_model()

# Example usage with new texts tailored to different classes
new_examples = [
    "Improve the system's performance by optimizing the database queries and reducing latency.",
    "Implement advanced encryption methods to enhance security and prevent unauthorized access.",
    "Enhance usability by making the user interface more intuitive and accessible for all users.",
    "Ensure compatibility across multiple platforms, including mobile and desktop environments.",
    "Increase reliability by implementing robust error handling and failover mechanisms.",
    "Scalability improvements are needed to handle increased load efficiently.",
    "Implement automated tools to improve maintainability of the codebase.",
    "Ensure the software is portable across different operating systems.",
    "Evaluate interoperability with other systems and ensure seamless integration."
]

# Run predictions to show different classes
predicted_labels = predict_new_examples(model, tokenizer, new_examples)
save_predictions(new_examples, predicted_labels)
