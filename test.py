from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 1. Load the trained model
# Load the saved model and tokenizer
tokenizer = BertTokenizer.from_pretrained("./mon_modele_bert_finetuned")
model = BertForSequenceClassification.from_pretrained("./mon_modele_bert_finetuned")

# 2. Function to predict sentiment for a single text
def predict_sentiment(text):
    # Set model to evaluation mode
    model.eval()
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", max_length=512, padding="max_length", truncation=True)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
    
    # For IMDB: 0 = negative, 1 = positive
    sentiment = "positive" if predicted_class == 1 else "negative"
    confidence = predictions[0][predicted_class].item()
    
    return {
        "sentiment": sentiment,
        "class": predicted_class,
        "confidence": confidence
    }

# 3. Test the function with example texts
examples = [
    "This movie was excellent, I loved every minute of it.",
    "What a waste of time, I regret watching this movie."
]

for example in examples:
    result = predict_sentiment(example)
    print(f"Text: {example}")
    print(f"Predicted Sentiment: {result['sentiment']} (confidence: {result['confidence']:.2f})")
    print("-" * 50)

# 4. Evaluate on a larger test dataset
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader

def evaluate_on_test_set(test_dataset, batch_size=32):
    # Create a DataLoader
    dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Prepare lists to store predictions and true labels
    all_predictions = []
    all_labels = []
    
    # Set model to evaluation mode
    model.eval()
    
    # Make batch predictions
    with torch.no_grad():
        for batch in dataloader:
            # Retrieve inputs and labels
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to('cpu').numpy()
            
            # Make predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).to('cpu').numpy()
            
            # Store predictions and labels
            all_predictions.extend(predictions)
            all_labels.extend(labels)
    
    # Compute evaluation metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions, target_names=["negative", "positive"])
    
    return {
        "accuracy": accuracy,
        "report": report
    }

# To use this evaluation function, you can create a new test dataset
# or use an existing one:
# results = evaluate_on_test_set(tokenized_datasets["test"])
# print(f"Accuracy: {results['accuracy']}")
# print(results['report'])

# 5. Perform inference on new data (e.g., from a file)
def predict_from_file(file_path):
    # Read texts from a file
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
    
    results = []
    for text in texts:
        if text:  # Ignore empty lines
            prediction = predict_sentiment(text)
            results.append((text, prediction))
    
    return results

# Example usage:
# predictions = predict_from_file("my_texts.txt")
# for text, prediction in predictions:
#     print(f"Text: {text[:50]}...")
#     print(f"Sentiment: {prediction['sentiment']} (confidence: {prediction['confidence']:.2f})")
