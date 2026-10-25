# Day 11: LSTM for Text Classification

## Project Overview
This project builds a **Long Short-Term Memory (LSTM)** neural network to classify SMS messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and Deep Learning.

The project reuses the tokenization and padding approach from Day 10, replacing the SimpleRNN layer with an LSTM layer to explore text classification using recurrent neural networks.

## Objective
- Classify SMS messages into spam and ham categories.
- Perform data cleaning and exploratory data analysis (EDA).
- Engineer meaningful text-based features.
- Build and train an LSTM model for 5 epochs.
- Evaluate model performance using test accuracy.

## Dataset
**Dataset:** `spam.csv` (SMS Spam Collection)

The dataset contains SMS messages and their corresponding labels.

| Column | Description |
|---|---|
| `v1` | Message label (ham or spam) |
| `v2` | SMS message text |

The original CSV also contains additional unnamed columns, which are excluded during preprocessing.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- Jupyter Notebook / Google Colab

## Project Workflow

### 1. Data Cleaning
- Loaded the SMS dataset.
- Selected the required label and text columns.
- Checked missing values and duplicate records.
- Removed duplicate messages.
- Encoded labels: Ham = 0, Spam = 1.

### 2. Exploratory Data Analysis (EDA)
Performed data analysis using:
- Spam vs. ham distribution.
- Message length histogram.
- Message length boxplot by label.
- Currency symbol association heatmap.
- Group comparisons of text characteristics.

### 3. Feature Engineering
Created the following features:
- `text_length`: Number of characters in a message.
- `word_count`: Number of words.
- `has_currency_symbol`: Indicates whether a currency symbol is present.
- `uppercase_ratio`: Proportion of uppercase alphabetic characters.

Message-length outliers were investigated and retained because extreme lengths do not necessarily indicate invalid messages.

### 4. Text Preprocessing
- Split the dataset into training and testing sets.
- Used an 80/20 train-test split.
- Applied Keras Tokenizer with a vocabulary limit of 5,000 words.
- Converted messages into numerical sequences.
- Used `pad_sequences(maxlen=50)` to ensure every message has 50 tokens.
- Fitted the tokenizer only on the training data.

### 5. LSTM Model Architecture
The following architecture was implemented:

`Embedding(5000, 32) → LSTM(32) → Dense(1, activation="sigmoid")`

| Layer | Configuration |
|---|---|
| Embedding | 5,000 vocabulary size, 32 dimensions |
| LSTM | 32 units |
| Dense | 1 neuron, sigmoid activation |

**Compilation settings:**
- Optimizer: Adam
- Loss: Binary Crossentropy
- Metric: Accuracy
- Epochs: 5
- Batch size: 32

### 6. Model Evaluation
The model was evaluated using the held-out test dataset.

Evaluation metrics:
- Test Accuracy
- Test Loss
- Precision
- Recall
- F1-score
- Confusion Matrix

Training and validation accuracy were also visualized.

## Results

| Metric | Result |
|---|---|
| Test Accuracy | Add your actual result |
| Test Loss | Add your actual result |
| Epochs | 5 |

*Results should be updated using the actual notebook output.*

## Key Learnings
- Learned how to preprocess text for recurrent neural networks.
- Understood how tokenization and padding work.
- Explored LSTM-based text classification.
- Practiced feature engineering and EDA for text data.
- Evaluated classification performance using multiple metrics.

## What Is an LSTM Forget Gate?
The forget gate allows an LSTM to decide which information from its previous cell state to retain or discard. This helps the model manage long-term dependencies and mitigate vanishing gradient problems more effectively than a plain RNN.

## Conclusion
This project demonstrates how an LSTM model can be used for SMS spam detection. It combines data preprocessing, feature engineering, exploratory analysis, deep learning, and model evaluation in a complete text classification workflow.

## Author
**Chamanthi Karthik**

Machine Learning | Data Analysis | Python
