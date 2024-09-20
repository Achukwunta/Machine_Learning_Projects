# Fake News Detection Analysis

## Table of Contents


- [INTRODUCTION](#introduction)
- [METHODOLOGY](#methodology)
- [RESULTS](#results)
- [RECOMMENDATIONS](#recommendations)
- [CONCLUSION](#conclusion)

### INTRODUCTION

Leveraging advanced natural language processing (NLP) techniques, this project aims to build a system to identify fake and real news articles using Term Frequency-Inverse Document Frequency (TF-IDF) and Logistic Regression model for classification.
The dataset used in this project consists of the following columns:

- **id**: Unique identifier for each news article
- **title**: Title of the news article
- **author**: Author of the news article
- **text**: Content of the news article
- **label**: Binary label indicating whether the news article is fake (1) or real (0)

### METHODOLOGY
#### Data Collection and Preprocessing:

The dataset, sourced from **Kaggle.com** in CSV format, was imported and processed within a Jupyter Notebook environment. During the data preprocessing phase, a custom function named **clean_text()** was defined to handle text cleaning tasks. This involved converting text to lowercase, eliminating special characters, punctuation marks, numbers, and stopwords, followed by word stemming.
Subsequently, the **clean_text()** function was applied to both the 'title' and 'text' columns of the DataFrame. The resulting cleaned text was stored in two new columns named 'clean_title' and 'clean_text', respectively.

#### Exploratory Data Analysis (EDA):

The distribution of labels was visualized using a count plot to provide insights into the distribution of the target variable across the dataset.

#### Feature Extraction:

The **TF-IDF vectorizer** was initialized with a maximum of 5000 features to limit the dimensionality of the feature space. It was then fitted on the cleaned text data to learn the vocabulary and IDF weights, capturing the significance of each term across the corpus.
Subsequently, the cleaned text data was transformed into TF-IDF features using the fitted vectorizer, representing each document as a vector of TF-IDF values. Finally, the TF-IDF features were converted into a DataFrame suitable for model training, enabling the application of machine learning algorithms on the transformed textual data.

#### Model Building:

The TF-IDF features (stored in tfidf_df) and the corresponding labels (dataframe['label']) were split into training and testing sets using the train_test_split function. Afterward, a Logistic Regression model was initialized and trained on the training data (X_train and y_train). Subsequently, the trained model was utilized to make predictions on the testing set (X_test).
To assess the model's performance, various metrics were computed including accuracy and a classification report, which contains precision, recall, F1-score, and support for each class.
Additionally, a confusion matrix was plotted to visually represent the distribution of true negative, false positive, false negative, and true positive values, providing further insight into the model's performance.

### RESULTS
#### Accuracy:
The model achieved an accuracy of approximately 94.09%, indicating that it correctly classified 94.09% of the news articles. This means that the model performs well on the test set.

#### Precision, Recall, and F1-Score:
- **Precision**: A precision score of 0.94 for label 0 means that 94% of the articles predicted as real news are actually real. Similarly, a precision score of 0.94 for label 1 means that 94% of the articles predicted as fake news are indeed fake.
- **Recall**: A recall score of 0.96 for label 0 means that the model correctly identified 96% of the real news articles. Similarly, a recall score of 0.92 for label 1 means that the model identified 92% of the fake news articles.

- **F1-Score**: The weighted average F1-score of approximately 0.94 indicates the overall performance of the model in terms of both precision and recall.

#### Support: 
There are 2600 instances of real news (label 0) and 1972 instances of fake news (label 1).
#### Confusion Matrix: 
The confusion matrix revealed that the model predicted 1815 true positives and 2487 true negatives. However, it also predicted 113 false positives and 157 false negatives.

### RECOMMENDATIONS
- Fine-tune the model hyperparameters or consider using more sophisticated algorithms to potentially improve performance further
- Conducting error analysis on misclassified instances can provide insights into the model's weaknesses and guide future improvements

### CONCLUSION

New language processing (NLP) was employed to construct a predictive model. The dataset was thoroughly explored, key aspects were visualized, and classification models were developed and assessed. Notably, the Logistic Regression exhibited commendable accuracy in predicting news article as fake or real. This project holds a significant value for users to make informed decisions and discern fact from fiction in an era of rampant misinformation.
It's important to acknowledge that while this serves as a simplified illustration, real-world applications often entail more intricate data preprocessing and feature engineering methodologies.

















