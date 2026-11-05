* NLP-Based Resume–Job Matching Classification

An exploratory natural language processing (NLP) project that classifies resume–job pairs as Match or No Match using text from candidate resumes and job requirements. The project compares TF-IDF-based machine learning with an LSTM neural network.
Scope: This is an academic experiment on a dataset-derived matching label, not a validated hiring recommendation tool. Automated predictions should not be used to make employment decisions without rigorous validation, fairness assessment, and human oversight.

Project Objective
Investigate whether NLP models can predict whether a resume–job pair meets a matching threshold, and compare the performance of:
- Logistic Regression with TF-IDF features
- Random Forest with TF-IDF features
- LSTM with tokenized text sequences
The binary target is derived from the dataset's existing matched_score:
- Match (1): matched_score >= 0.7
- No Match (0): matched_score < 0.7
This target represents a thresholded dataset score, rather than an independently verified recruiter judgment or hiring outcome.
Dataset
Source: Kaggle — Resume Dataset

The notebook reads resume_data.csv and uses text describing resume qualifications and job requirements, including career objectives, skills, education, professional experience, job positions, responsibilities, and required skills.
Characteristic	Value
Resume–job pairs	9,544
Original columns	35
No Match (0)	5,075 (53.17%)
Match (1)	4,469 (46.83%)
Training set	7,635 (80%)
Test set	1,909 (20%)
Target threshold	0.70


The dataset is not included in this repository. Download it from the source above to reproduce the experiment.
Technologies
- Python — pandas, NumPy
- Text preprocessing — NLTK tokenization, stopword removal, WordNet lemmatization
- Classical NLP / ML — scikit-learn, TF-IDF, Logistic Regression, Random Forest
- Hyperparameter tuning — RandomizedSearchCV
- Deep learning — TensorFlow / Keras (Embedding, LSTM, Dropout, Dense)
- Exploration and visualization — Matplotlib, Seaborn, WordCloud
- Notebook environment — Google Colab with Google Drive
Methodology
1. Data exploration: Inspect dataset dimensions, null values, matching-score distribution, class balance, common skills, and text length.
2. Missing-value handling: Replace missing text values with Not Available.
3. Text preprocessing: Lowercase text, tokenize, keep alphabetic tokens, remove English stopwords, and lemmatize.
4. Feature construction: Concatenate selected candidate- and job-related text into combined_text.
5. Classical models: Vectorize combined text using TF-IDF with up to 5,000 unigram/bigram features; train Logistic Regression and Random Forest.
6. Random Forest tuning: Run randomized hyperparameter search with three-fold cross-validation.
7. LSTM: Tokenize the training text, pad/truncate sequences to 200 tokens, and train an Embedding → LSTM → Dropout → Dense classifier with early stopping.
8. Evaluation: Compare accuracy, Match-class precision, recall, F1-score, and confusion matrices; inspect Random Forest feature importance and prediction errors.
An 80/20 stratified random split (random_state=42) is used for the reported model results.
Results
The following figures are from the saved outputs in Resume-Job Matching.ipynb, on the 1,909-record test partition. Precision, recall, and F1-score are for Match (1).
Model	Accuracy	Precision	Recall	F1-score
Logistic Regression	75.12%	72.31%	75.95%	74.09%
Random Forest	77.74%	75.30%	78.08%	76.66%
LSTM	55.89%	77.08%	8.28%	14.95%


Key Findings
- Random Forest achieved the highest test accuracy and F1-score among the three main models in this experiment.
- Logistic Regression provided a competitive, simpler baseline.
- LSTM had very low recall for the Match class (8.28%), missing most positive examples despite relatively high precision among the few positive predictions it made.
- The results suggest that TF-IDF-based approaches were more effective than the tested LSTM configuration on this particular task. This does not establish that traditional methods outperform deep learning in general.
Random Forest Hyperparameter Search
The notebook also tested Random Forest configurations using RandomizedSearchCV (five sampled configurations, three-fold CV). Its reported best configuration was:
{
    "n_estimators": 150,
    "max_depth": 20,
    "min_samples_split": 2,
    "min_samples_leaf": 1
}
- Best cross-validation F1-score: 0.7689
- Tuned Random Forest test accuracy: 76.22%
- Tuned Random Forest test F1-score: 75.83%
The tuned configuration did not outperform the original Random Forest on the reported test F1-score. The main comparison table therefore reports the original model.
Feature Importance and Error Analysis
The Random Forest analysis highlights influential TF-IDF features, including terms such as name, year, learning, management, compliance, and analytics. These are model-specific associations and should not be interpreted as causal determinants of candidate suitability.

The notebook also inspects incorrectly classified resume–job pairs. These examples can help investigate vocabulary differences, missing context, and errors in the dataset-derived labels.
Repository Files
File	Description
Resume-Job Matching.ipynb	Data exploration, preprocessing, model training, evaluation, and visualizations
Resume-Job Matching Report.pdf	University project report
Resume-Job Matching Presentation.pdf	University project presentation
README.md	Project overview, findings, and limitations
.gitignore	Git ignore rules


Running the Notebook
The existing notebook was developed in Google Colab, using a dataset saved in Google Drive.
1. Download resume_data.csv from the linked Kaggle dataset.
2. Upload the CSV to the following path in Google Drive: MyDrive/NLP-project/resume_data.csv.
3. Open Resume-Job Matching.ipynb in Google Colab.
4. Install any dependencies not already available in the runtime. The notebook uses pandas, numpy, nltk, scikit-learn, tensorflow, matplotlib, seaborn, and wordcloud.
5. Authorize Google Drive access when prompted.
Reproducibility note: The notebook is preserved in its original exploratory form. It has execution-order dependencies (including a cell that references combined_text before its later construction), and its package versions are not pinned. A clean, top-to-bottom, reproducible workflow remains a future improvement; the saved metrics above have not been revalidated in a fresh run.
Limitations and Future Improvements
- Data-derived labels: The target comes from a threshold applied to matched_score, not observed recruitment outcomes or independent human review. Results measure prediction of that thresholded score, not hiring quality.
- Potential data leakage: TF-IDF is fitted to the entire text dataset before the classical models' train/test split. In a revised pipeline, the vectorizer should be fitted on the training data only (and within cross-validation folds for tuning). Reported results should be re-evaluated after this correction.
- Generalization: A single random split does not establish performance on new job postings, employers, candidate populations, or time periods. Future tests should assess possible overlap/duplicates and use appropriately separated evaluation sets.
- Architecture and training: Further LSTM tuning, better handling of long text, and a fairer comparison with other embedding approaches could be investigated. BERT was considered but was not trained and evaluated in this notebook.
- Fairness and privacy: Resume data can contain personal and potentially sensitive information. Deployment would require privacy safeguards, bias testing, explainability, and human oversight, rather than automatic screening decisions.
- Reproducibility: Refactor notebook execution order; pin package versions; make the dataset path configurable; retain an untouched test set; and add reproducible training instructions.
These changes are recommendations for future work and are not yet implemented in the published notebook.
Conclusion
On the notebook's reported test split, Random Forest achieved the strongest overall results (77.74% accuracy, 76.66% F1-score), followed by Logistic Regression. The tested LSTM substantially underperformed on Match-class recall. The project demonstrates an end-to-end NLP classification workflow and illustrates why careful evaluation, leakage prevention, and responsible use are essential when working with employment-related data.
