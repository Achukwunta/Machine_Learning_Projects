## Zomato Restaurant Data Analysis: Predicting Success Rate

### Table of Contents
- [INTRODUCTION](#introduction)
- [METHODOLOGY](#methodology)
- [RESULTS](#results)
- [RECOMMENDATIONS](#recommendations)
- [CONCLUSION](#conclusion)

### INTRODUCTION
Zomato, a leading online platform for food delivery and restaurant discovery, has revolutionized the global culinary landscape. Its comprehensive dataset offers invaluable insights into restaurants' offerings, pricing, and ratings, catering to a diverse range of food enthusiasts and budget-conscious diners. 

In this data analysis endeavor, my objective is to leverage the richness of Zomato's data to build a model that predicts the restaurants that will have success rate.

### METHODOLOGY
- **Data Collection and Preprocessing:**
The Zomato dataset was collected from Kaggle.com in SQL format and the data was accessed by connecting to the SQLite database using Jupyter Notebook. During preprocessing, the focus was on data cleaning, particularly fixing formatting issues in two features: "rate" and "approx_cost(for two people)". Missing values and outliers were deliberately not handled at this stage to avoid losing significant data for exploratory data analysis (EDA).

- **Exploratory Data Analysis (EDA):**
During the Exploratory Data Analysis (EDA) phase, the relationships and associations between the 'rate', 'approx_cost(for two people)', and 'votes' features were visualized using distplot and heatmap visualizations. Additionally, correlation coefficients were calculated to quantify these relationships. Furthermore, visualizations were employed to highlight insights, such as identifying the top restaurant (Quick Bites) with frequently ordered dishes (most liked dishes in Quick Bites) and determining the most ordered dishes across all restaurants (most liked dishes across restaurants) using word clouds.

- **Feature Engineering:**
During feature engineering, missing values were addressed in important features before selecting relevant features using distribution plots. A binary target variable was created, categorizing restaurants with a rating threshold of >=3.5 as 'good' and others as 'bad'. Categorical features were encoded using one-hot encoding for the top 10 categories covering up to 80% of the data, while mean encoding was utilized for categories not meeting this criterion.
Moreover, numerical features were scaled to extract useful information from text data.

- **Model Selection:**
During model selection, outliers were detected and addressed through log transformation techniques. The dataset was transformed accordingly. Subsequently, the data was split into training and testing sets to enable model evaluation. For the prediction task, random forests were employed as the chosen machine learning algorithm.

- **Model Training and Evaluation:**
During model training and evaluation, the data was split into training and testing sets. Models were trained on the training data and evaluated using appropriate metrics such as accuracy, precision, recall, and F1-score.

### RESULTS
- **Accuracy:** the model achieved an accuracy of approximately 89.27%, which was a good score and showed high accuracy and reliability. This suggests that the model performs well on the test set.
- **Precision, Recall, and F1-Score:**
Precision, recall, and F1-score were calculated for each class separately 
- **Precision:** For class 0, the precision was 84%, which suggested that the model predicted an instance as class 0, it was correct about 84% of the time. For class 1, the precision was higher at 91%, which indicated that the model has a higher precision for predicting class 1 instances.
- **Recall:** For class 0, the recall was 75%, which implied that the model correctly identified 75% of all actual class 0 instances. For class 1, the recall was higher at 95%, which indicated that the model effectively captures a higher proportion of actual class 1 instances.
- **F1-Score:** the F1-score for class 0 is 0.79, and for class 1, it was 0.93, which suggested that the model achieved a better balance between precision and recall for class 1.

- **Support:** Support values (2164 for class 0 and 7241 for class 1) showed the distribution of each class in the test data.
- **Confusion Matrix:** the confusion matrix revealed that the model predicts 2164 true positives and 7241 true negatives. However, it also predicts 732 false positives and 399 false negatives.

### RECOMMENDATIONS
- Fine-tune the model hyperparameters or consider using more sophisticated algorithms to potentially improve performance further.
- Investigate the specific characteristics of misclassified instances to gain insights into areas for model improvement.

### CONCLUSION
Data analysis and machine learning techniques were employed to construct a predictive model for determining successful restaurants. The dataset was thoroughly explored, key aspects were visualized, and classification models were developed and assessed. Notably, the Random Forest Classifier exhibited commendable accuracy in predicting success rates. Such a model holds significant value for restaurant owners and other businesses in facilitating data-driven decision-making processes. It's important to acknowledge that while this serves as a simplified illustration, real-world applications often entail more intricate data preprocessing and feature engineering methodologies.
