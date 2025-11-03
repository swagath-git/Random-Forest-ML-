# Mushroom Classification using Random Forest (Machine Learning Project)
# Project Overview
    This project aims to build a predictive model that determines whether a mushroom is edible or poisonous based on its physical characteristics. 
    The solution utilizes the Random Forest Classifier, a powerful ensemble learning method, to achieve high accuracy and interpretability.

# Objectives
    1. Develop a machine learning model to classify mushrooms as edible or poisonous.
    2. Identify key features influencing mushroom toxicity.
    3. Provide insights that can help people identify poisonous mushrooms based on their attributes.

# Workflow
  # 1. Data Preprocessing
     (i). Handled missing or inconsistent data.
     (ii). Encoded categorical features for model compatibility.
     (iii). Split the dataset into training and testing sets for evaluation.

  # 2. Feature Engineering
     (i). Selected relevant attributes that influence mushroom classification.
    (ii). Analyzed correlations and feature importance scores.

# 3. Model Building
     (i). Implemented a Random Forest Classifier from scikit-learn.
    (ii). Trained and evaluated the model using accuracy, precision, recall, and F1-score metrics.

# 4. Hyperparameter Tuning
Optimized parameters such as:
    (i). n_estimators (number of trees)
     (ii). max_depth
    (iii). min_samples_split
     (iv)min_samples_leaf
 Used GridSearchCV for fine-tuning the model to achieve the best performance.

# 5. Model Evaluation
    1. Compared model performance before and after tuning.
    2. Generated a confusion matrix and classification report.
    3. Plotted feature importance to interpret which features most affect predictions.

# Results & Insights
    1. The optimized Random Forest model achieved high accuracy on test data.
    2. Certain visual and structural features of mushrooms strongly influence their classification.
    3. The model helps illustrate how machine learning can be used in biological classification and safety prediction.

# Technologies Used
    1. Python
    2. Pandas, NumPy — Data manipulation and preprocessing
    3. Matplotlib, Seaborn — Data visualization
    4. Scikit-learn — Machine learning model and evaluation
