# Amazon Review Sentiment Classifier

This project applies Natural Language Processing (NLP) techniques to classify Amazon food reviews into three sentiment categories: **Positive**, **Neutral**, and **Negative**. It includes both traditional vectorization methods and modern transformer-based models.

## Dataset

The dataset used contains Amazon food reviews and is available publicly on Kaggle. It includes review texts and their associated sentiment labels.

## Objective

The goal is to build a **multi-class classifier** that predicts sentiment based on review text. The task is framed as a supervised learning problem with three target classes.

## Approach

Several techniques were used and compared:

- **Text Vectorization**:  
  - CountVectorizer and TF-IDF using Scikit-learn

- **Word Embeddings**:  
  - Pre-trained embeddings like GloVe, Word2Vec, and FastText with Gensim

- **Dimensionality Reduction & Visualization**:  
  - PCA and t-SNE for vector visualization

- **Transformer-based Models**:  
  - DistilBERT embeddings via Hugging Face  
  - Fine-tuned DistilBERT model using PyTorch

## Results

| Method                                 | F1 Score (Macro Avg.) |
|----------------------------------------|------------------------|
| GloVe + SGD                            | 0.46                   |
| DistilBERT Embeddings + SGD            | 0.566                  |
| DistilBERT Embeddings + Logistic Stack | 0.69                   |
| Fine-tuned DistilBERT                  | **0.75**               |

## Tools & Libraries

- Python  
- Scikit-learn  
- PyTorch  
- Hugging Face Transformers  
- Gensim  
- Matplotlib / Seaborn  
- SHAP (for model interpretability)

## Author

Abdelrahman Abdelgawad  
M.Eng. Data Science — TU Darmstadt  
📧 abdelrahmansaeed291@gmail.com

