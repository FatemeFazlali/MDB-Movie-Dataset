# Movie Profitability Prediction and Recommendation System

This project aims to predict the profitability of movies before their release and create a recommendation system based on movie features. The project includes Exploratory Data Analysis (EDA), Data Modeling with machine learning algorithms, and the implementation of a recommendation system using TF-IDF vectorization.

## Overview

The project consists of the following main components:

1. **Exploratory Data Analysis (EDA)**: Gain insights into the dataset and understand the relationships between various features and movie profitability.

2. **Data Modeling**:
   - Train different machine learning models (Logistic Regression, Decision Tree, Random Forest, XGBoost) to predict movie profitability.
   - Evaluate model performance using metrics like accuracy, precision, recall, F1-Score, and ROC AUC Score.

3. **Recommendation System**:
   - Implement a recommendation system based on TF-IDF vectorization.
   - Calculate cosine similarity between movie vectors and recommend similar movies based on user input.

## Usage

1. **Exploratory Data Analysis**:
   - Execute the EDA.ipynb file to perform exploratory data analysis on the provided dataset.
   - Gain insights into movie profitability, budget vs. revenue relationship, genre frequencies, director and actor impact, and release date trends.

2. **Data Modeling**:
   - Execute the Data_Modeling.ipynb file to preprocess the data, train machine learning models, and evaluate their performance.
   - Perform feature engineering, split the dataset into training and testing sets, and train various models.
   - Evaluate model performance using accuracy, precision, recall, F1-Score, and ROC AUC Score.

3. **Recommendation System**:
   - Execute the Recommendation_System.ipynb file to implement the recommendation system.
   - Utilize TF-IDF vectorization to create movie vectors based on keywords.
   - Calculate cosine similarity between movie vectors and recommend similar movies based on user input.

## Additional Notes

- The project dataset contains information on movie features such as budget, genres, cast, crew, and revenue.
- Hyperparameter tuning is performed for each machine learning model using GridSearchCV.
- Experiment with different machine learning algorithms and hyperparameters to find the best model for predicting movie profitability.
- The recommendation system suggests similar movies based on TF-IDF vectorization and cosine similarity.

## Requirements

- Python 3.x
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

## Credits

This project is developed by Fateme Fazlali

## License

This project is licensed under the [MIT License](LICENSE).
