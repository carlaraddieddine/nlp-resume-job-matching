# NLP-Based Resume-Job Matching Classification

## Project Overview
This project focuses on building an NLP-based classification system to determine whether a candidate resume matches a job description. The model classifies each resume-job pair as either Match or No Match based on textual information.

## Objective
The goal of this project is to automate resume-job matching using Natural Language Processing and machine learning techniques.

The original dataset included a matched score, which was converted into a binary classification target using a threshold of 0.7:
- Match: score >= 0.7
- No Match: score < 0.7

## Dataset
The dataset was obtained from Kaggle and contains resume-job pairs with resume-related and job-related features.

Dataset link: https://www.kaggle.com/datasets/saugataroyarghya/resume-dataset

The dataset includes:
- Resume skills
- Education information
- Job titles
- Required skills
- Responsibilities
- Experience requirements
- Matched score

## Tools and Libraries
- Python
- pandas
- NumPy
- scikit-learn
- NLTK
- TensorFlow/Keras
- Matplotlib
- Seaborn

## Models Implemented
- Logistic Regression
- Random Forest
- LSTM

## Methodology
- Handled missing text values
- Cleaned and normalized text
- Applied lowercasing, tokenization, stopword removal, and lemmatization
- Combined resume and job-related text fields into one feature
- Used TF-IDF for classical machine learning models
- Used tokenized and padded sequences for the LSTM model
- Trained and compared multiple models
- Evaluated performance using classification metrics

## Evaluation Metrics
The models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Key Results
- Random Forest achieved the best overall performance.
- Logistic Regression performed well as a strong baseline model.
- LSTM underperformed compared to the classical machine learning models.
- TF-IDF was effective for this keyword-driven resume-job matching task.

## Ethical Considerations
This project highlights that automated resume screening systems should be used as decision-support tools, not as fully automated hiring systems. Human oversight is important because keyword-based matching may introduce bias or disadvantage candidates who describe similar skills using different wording.

## Files
- `Resume-Job Matching.ipynb`: Main notebook implementation
- `Resume-Job Matching Report.pdf`: Full project report
- `Resume-Job Matching Presentation.pdf`: Project presentation

## Conclusion
This project shows that traditional NLP and machine learning methods can perform well for resume-job matching when the task depends strongly on keyword overlap and structured textual features.
