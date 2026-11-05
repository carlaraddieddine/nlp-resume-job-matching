
# NLP-Based Resume–Job Matching Classification

An exploratory Natural Language Processing (NLP) project that classifies resume–job pairs as **Match** or **No Match** using candidate qualifications and job requirements.

The project compares traditional machine learning models using TF-IDF features with an LSTM neural network.

> **Note:** This is an academic machine learning project, not a validated hiring recommendation system. Automated predictions should not be used to make employment decisions without rigorous validation, fairness assessment, and human oversight.

## 1. Project Overview

Resume screening can be a time-consuming process, particularly when organizations receive large numbers of applications.

Natural Language Processing offers techniques for analyzing textual information and identifying similarities between candidate qualifications and job requirements.

This project investigates whether machine learning models can predict whether a resume–job pair meets a predefined matching threshold.

### Objectives

- Develop a binary classification system for resume–job matching.
- Apply NLP preprocessing techniques to resume and job-related text.
- Compare traditional machine learning models with a deep learning approach.
- Evaluate model performance using multiple classification metrics.
- Analyze feature importance and model prediction errors.
- Identify limitations and potential improvements.

The target variable is derived from the dataset's existing `matched_score`:

- **Match (1):** `matched_score >= 0.7`
- **No Match (0):** `matched_score < 0.7`

This target represents a thresholded dataset score rather than an independently verified hiring decision.

## 2. Dataset

**Source:** [Kaggle — Resume Dataset](https://www.kaggle.com/datasets/saugataroyarghya/resume-dataset)

The dataset contains candidate resume information and corresponding job requirements.

Relevant information includes:

- Candidate skills
- Educational background
- Professional experience
- Career objectives
- Job titles and positions
- Job responsibilities
- Required qualifications and skills
- Resume–job matching scores

### Dataset Characteristics

| Characteristic | Value |
|---|---:|
| Total resume–job pairs | 9,544 |
| Original features | 35 |
| No Match (0) | 5,075 (53.17%) |
| Match (1) | 4,469 (46.83%) |
| Training set | 7,635 (80%) |
| Test set | 1,909 (20%) |
| Matching threshold | 0.70 |

The original dataset is not included in this repository and can be downloaded from Kaggle.

## 3. Technologies and Libraries

**Programming Language**
- Python

**Data Processing**
- pandas
- NumPy

**Natural Language Processing**
- NLTK
- TF-IDF Vectorization
- Tokenization
- Stopword Removal
- Lemmatization

**Machine Learning**
- scikit-learn
- Logistic Regression
- Random Forest
- RandomizedSearchCV

**Deep Learning**
- TensorFlow
- Keras
- Embedding Layers
- LSTM

**Data Visualization**
- Matplotlib
- Seaborn
- WordCloud

**Development Environment**
- Google Colab
- Google Drive

## 4. Methodology

### 4.1 Data Exploration

Exploratory Data Analysis (EDA) was performed to understand the dataset and identify relevant patterns.

The analysis included:

- Dataset dimensions and feature types
- Missing values
- Matching-score distribution
- Target class distribution
- Common skills and keywords
- Text length analysis

### 4.2 Text Preprocessing

Text preprocessing was applied to normalize the resume and job-related information.

The steps included:

1. Handling missing text values.
2. Converting text to lowercase.
3. Tokenizing text into individual words.
4. Removing English stopwords.
5. Removing non-alphabetic tokens.
6. Applying WordNet lemmatization.
7. Combining selected resume and job-related text fields.

The resulting `combined_text` feature was used as input for the classification models.

### 4.3 Feature Extraction

Two approaches were used to represent textual information.

**TF-IDF Vectorization**

TF-IDF was used for the traditional machine learning models.

The vectorizer included:
- Unigrams and bigrams
- A maximum of 5,000 features

**Tokenized Sequences**

For the LSTM model, text was converted into integer sequences and padded or truncated to a maximum sequence length of 200 tokens.

### 4.4 Train-Test Split

The dataset was divided into:

- 80% training data
- 20% testing data

A stratified random split was used with `random_state=42` to preserve the class distribution.

A limitation of the original implementation is that TF-IDF vectorization was performed before the train-test split. This introduces potential data leakage and should be corrected in future experiments.

## 5. Models Implemented

Three machine learning approaches were evaluated.

### 5.1 Logistic Regression

Logistic Regression was used as a baseline classification model.

It provides an interpretable and computationally efficient approach for learning relationships between TF-IDF features and matching labels.

### 5.2 Random Forest

Random Forest combines multiple decision trees to generate classification predictions.

It was used to capture nonlinear relationships between textual features and the matching target.

Additional experiments included:
- Hyperparameter tuning
- Cross-validation
- Feature importance analysis
- Prediction error analysis

### 5.3 Long Short-Term Memory (LSTM)

An LSTM neural network was implemented to investigate whether a sequential deep learning model could improve resume–job classification.

The architecture included:
- Embedding layer
- LSTM layer
- Dropout layer
- Dense classification layer

Early stopping was used during training.

## 6. Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Precision, recall, and F1-score refer to the positive **Match (1)** class.

### 6.1 Model Performance Comparison

The following results were obtained from the saved notebook outputs.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 75.12% | 72.31% | 75.95% | 74.09% |
| **Random Forest** | **77.74%** | 75.30% | **78.08%** | **76.66%** |
| LSTM | 55.89% | **77.08%** | 8.28% | 14.95% |

### 6.2 Key Findings

**Random Forest**

Random Forest achieved the strongest overall performance among the evaluated models.

It obtained:
- Accuracy: 77.74%
- Precision: 75.30%
- Recall: 78.08%
- F1-score: 76.66%

**Logistic Regression**

Logistic Regression provided a competitive baseline, achieving an F1-score of 74.09%.

Its performance demonstrates the effectiveness of traditional classification methods when combined with TF-IDF features.

**LSTM**

The LSTM model achieved significantly lower recall than the traditional machine learning models.

Although its precision reached 77.08%, recall was only 8.28%, meaning the model failed to identify most positive matches.

This suggests that the tested LSTM configuration was less effective for this particular classification task.

These findings do not imply that traditional machine learning generally outperforms deep learning. They apply to the dataset, preprocessing methods, and model configurations tested in this project.

## 7. Random Forest Hyperparameter Tuning

Hyperparameter tuning was performed using `RandomizedSearchCV`.

Five parameter configurations were evaluated using three-fold cross-validation.

### Best Parameters

- `n_estimators`: 150
- `max_depth`: 20
- `min_samples_split`: 2
- `min_samples_leaf`: 1

### Tuning Results

| Metric | Result |
|---|---:|
| Best Cross-Validation F1-score | 76.89% |
| Tuned Random Forest Accuracy | 76.22% |
| Tuned Random Forest F1-score | 75.83% |

The tuned Random Forest did not outperform the original Random Forest model on the reported test F1-score.

Therefore, the main performance comparison reports the original Random Forest results.

## 8. Feature Importance Analysis

Feature importance analysis was conducted using the Random Forest model.

Some influential TF-IDF features included:

- `name`
- `year`
- `learning`
- `management`
- `compliance`
- `analytics`

These features represent model-specific statistical associations rather than causal indicators of candidate suitability.

The notebook also includes an examination of incorrectly classified resume–job pairs to better understand prediction errors.

## 9. Ethical Considerations

Automated resume screening involves important ethical and practical considerations.

### Potential Bias

Keyword-based matching can disadvantage candidates who describe equivalent qualifications using different terminology.

Models may also reproduce biases present in their training data.

### Data Privacy

Resume information may include personal and potentially sensitive information.

Appropriate privacy and data protection measures are necessary before deploying such systems.

### Human Oversight

Automated resume–job matching should support recruitment professionals rather than independently determine employment decisions.

Predictions should be interpreted with caution, particularly when the underlying target is derived from an existing matching score rather than verified hiring outcomes.

## 10. Repository Structure

| File | Description |
|---|---|
| `Resume-Job Matching.ipynb` | Data exploration, NLP preprocessing, model training, evaluation, and visualizations |
| `Resume-Job Matching Report.pdf` | Full university project report |
| `Resume-Job Matching Presentation.pdf` | Project presentation |
| `README.md` | Project overview, methodology, results, and limitations |
| `.gitignore` | Git ignore configuration |

## 11. How to Run the Project

The original notebook was developed using Google Colab and Google Drive.

### Step 1 — Download the Dataset

Download the dataset from:

https://www.kaggle.com/datasets/saugataroyarghya/resume-dataset

### Step 2 — Prepare Google Drive

Upload `resume_data.csv` to:

`MyDrive/NLP-project/resume_data.csv`

### Step 3 — Open the Notebook

Open `Resume-Job Matching.ipynb` in Google Colab.

### Step 4 — Install Dependencies

The notebook uses the following libraries:

- pandas
- numpy
- nltk
- scikit-learn
- tensorflow
- matplotlib
- seaborn
- wordcloud

Install any dependencies that are unavailable in your environment.

### Step 5 — Run the Notebook

Authorize Google Drive access when prompted.

**Reproducibility note:** The notebook is preserved in its original exploratory form and contains execution-order dependencies. It also does not use pinned package versions.

A fully reproducible, top-to-bottom execution workflow remains a planned improvement. The saved results have not yet been independently revalidated through a fresh execution.

## 12. Limitations

This project is an academic NLP experiment and is not intended for production recruitment decisions.

The main limitations include:

**1. Dataset-Derived Labels**

The target variable was generated by applying a threshold to the existing matching score.

The results therefore measure the ability to predict that thresholded score, rather than independently verified candidate suitability.

**2. Potential Data Leakage**

TF-IDF vectorization was fitted to the entire dataset before the classical models' train-test split.

A revised implementation should fit the vectorizer using training data only.

**3. Generalization**

Performance was evaluated using a single random train-test split.

Additional evaluation is required to determine how well the models generalize to new job descriptions, candidate profiles, and recruitment contexts.

**4. LSTM Performance**

The tested LSTM configuration achieved low recall and may benefit from additional architecture and training experiments.

**5. Fairness and Explainability**

Further testing would be needed to assess demographic fairness, interpretability, and the reliability of predictions.

**6. Reproducibility**

The notebook would benefit from improved execution order, configurable file paths, and pinned library versions.

## 13. Future Improvements

Potential next steps include:

- Correcting the TF-IDF data leakage issue.
- Improving the notebook's reproducibility.
- Evaluating models using additional validation strategies.
- Investigating class imbalance and decision thresholds.
- Experimenting with alternative text representations.
- Further tuning the LSTM architecture.
- Comparing against pretrained transformer-based models.
- Investigating semantic similarity approaches.
- Conducting fairness and bias assessments.
- Developing an interpretable demonstration interface.

These improvements have not yet been implemented in the current notebook.

## 14. Conclusion

This project demonstrates the application of Natural Language Processing and machine learning techniques to resume–job matching classification.

Among the evaluated models, **Random Forest achieved the strongest overall performance**, with 77.74% accuracy and a 76.66% F1-score.

Logistic Regression provided a competitive baseline, while the tested LSTM architecture underperformed, particularly in identifying positive matches.

The project highlights the usefulness of traditional NLP representations such as TF-IDF for structured text classification tasks, while also emphasizing the importance of proper validation, data leakage prevention, and responsible AI practices.
