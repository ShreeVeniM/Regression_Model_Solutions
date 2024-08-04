**PROJECT OVERVIEW**


**Introduction**
This project focuses on predicting housing prices using various regression models, including Linear Regression, Decision Tree, and Random Forest. The analysis includes data loading, preprocessing, model training, evaluation, and saving/loading models.

**Directory Structure**
.git: Contains Git version control data.
.gitattributes: Git attributes file.
assets: Directory for storing assets such as images or other files.
main.py: The main script to run the project.
RE_Model: Directory for storing the trained Random Forest model.
requirements.text: Lists the Python dependencies.
src: Source code directory containing modules for various tasks.
tree.png: Visualization of the decision tree model.

**Key Components**
main.py
The central script that coordinates the following tasks:

**Data Loading:**
Loads housing price data from src/dataset/final.csv.

**Data Preprocessing:**
Splits the data into training and testing sets.

**Model Training and Evaluation:**
Trains and evaluates Linear Regression, Decision Tree, and Random Forest models.
Evaluates the models using Mean Absolute Error (MAE).
Prints evaluation results for each model.
Plots the decision tree and saves the plot.

**Saving and Loading Models:**
Saves the trained Random Forest model.
Loads the saved model and makes predictions.

**requirements.text**
Specifies the Python libraries required for the project:

pandas
numpy
matplotlib
scikit-learn

**Conclusion**
This project provides a comprehensive analysis and prediction of housing prices using Linear Regression, Decision Tree, and Random Forest models. It includes detailed evaluation metrics and the ability to save and load models for future predictions. ​
