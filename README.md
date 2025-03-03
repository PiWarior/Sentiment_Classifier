# Fine-tuning BERT for Sentiment Analysis

This project fine-tunes a **BERT** model for sentiment classification (positive/negative).

## 📌 Prerequisites

Ensure you have the following libraries installed:

```bash
pip install transformers torch scikit-learn datasets
```

## 📥 Installation & Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo.git
   cd your-repo
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download and prepare the dataset (example using IMDb):
   ```python
   from datasets import load_dataset
   dataset = load_dataset("imdb")
   ```

---

## 🎯 Fine-tuning the Model

The **BERT** model was fine-tuned on a sentiment dataset for 3 epochs.

| Epoch | Training Loss | Validation Loss | Accuracy | F1 Score | Precision | Recall |
|-------|--------------|----------------|----------|----------|-----------|--------|
| 1     | 0.226500     | 0.270268       | 89.46%   | 0.8851   | 0.9726    | 0.8120 |
| 2     | 0.125300     | 0.261332       | 93.29%   | 0.9311   | 0.9568    | 0.9068 |
| 3     | 0.052100     | 0.280319       | 94.02%   | 0.9404   | 0.9384    | 0.9424 |

### 📊 Evaluation Results:
```json
{
  "eval_loss": 0.2803,
  "eval_accuracy": 0.9402,
  "eval_f1": 0.9404,
  "eval_precision": 0.9384,
  "eval_recall": 0.9424,
  "eval_runtime": 793.55,
  "eval_samples_per_second": 31.50,
  "eval_steps_per_second": 0.49
}
```

The fine-tuned model is saved in the **mon_modele_bert_finetuned/** directory, containing:
- `tokenizer_config.json`
- `special_tokens_map.json`
- `vocab.txt`
- `added_tokens.json`

---

## 🚀 Testing the Model

After fine-tuning, test the model using `test.py`:

```bash
python test.py
```

Example output:
```
Text: This movie was excellent, I loved every minute of it.
Predicted Sentiment: Positive (Confidence: 1.00)
--------------------------------------------------
Text: What a waste of time, I regret watching this movie.
Predicted Sentiment: Negative (Confidence: 1.00)
--------------------------------------------------
```

---

## 📂 Project Structure

```
📁 sentiment_analysis_project
 ├── mon_modele_bert_finetuned/  # Fine-tuned model
 │   ├── tokenizer_config.json
 │   ├── special_tokens_map.json
 │   ├── vocab.txt
 │   ├── added_tokens.json
 ├── test.py                     # Model testing script
 ├── train.py                     # Training script
 ├── requirements.txt             # Dependencies
 ├── README.md                    # This file
```

---

## 📝 License

This project is licensed under the **MIT** License.

---

## ✨ Author

- **Your Name** - [GitHub](https://github.com/your-profile)

