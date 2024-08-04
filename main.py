import pandas as pd
import logging
from src.data_loader import load_data
from src.preprocess import split_data
from src.train import train_linear_regression, train_decision_tree, train_random_forest, save_model, load_model
from src.evaluate import evaluate_model, plot_tree
from src.utils import print_evaluation_results

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        file_path = 'src/dataset/final.csv'
        
        # Load and preprocess data
        logging.info("Loading data from final.csv")
        df = load_data(file_path)
        if df is None:
            return
        
        # Split data into training and testing sets
        logging.info("Splitting data into training and testing sets")
        x_train, x_test, y_train, y_test = split_data(df, target_variable='price')
        if x_train is None or x_test is None or y_train is None or y_test is None:
            return
        
        # Train and evaluate Linear Regression model
        logging.info("Training Linear Regression model")
        lr_model = train_linear_regression(x_train, y_train)
        if lr_model is not None:
            train_mae, test_mae = evaluate_model(lr_model, x_train, y_train, x_test, y_test)
            if train_mae is not None and test_mae is not None:
                logging.info("Evaluating Linear Regression model")
                print("Linear Regression Model:")
                print_evaluation_results(train_mae, test_mae)
        
        # Train and evaluate Decision Tree model
        logging.info("Training Decision Tree model")
        dt_model = train_decision_tree(x_train, y_train)
        if dt_model is not None:
            train_mae, test_mae = evaluate_model(dt_model, x_train, y_train, x_test, y_test)
            if train_mae is not None and test_mae is not None:
                logging.info("Evaluating Decision Tree model")
                print("Decision Tree Model:")
                print_evaluation_results(train_mae, test_mae)
                plot_tree(dt_model, 'charts/decision_tree.png')
        
        # Train and evaluate Random Forest model
        logging.info("Training Random Forest model")
        rf_model = train_random_forest(x_train, y_train)
        if rf_model is not None:
            train_mae, test_mae = evaluate_model(rf_model, x_train, y_train, x_test, y_test)
            if train_mae is not None and test_mae is not None:
                logging.info("Evaluating Random Forest model")
                print("Random Forest Model:")
                print_evaluation_results(train_mae, test_mae)
                save_model(rf_model, 'RE_Model')
        
        # Load and use the saved model
        logging.info("Loading saved Random Forest model")
        loaded_model = load_model('RE_Model')
        if loaded_model is not None:
            prediction = loaded_model.predict([[2012, 216, 74, 1, 1, 618, 2000, 600, 1, 0, 0, 6, 0]])
            logging.info(f"Loaded model prediction: {prediction}")
            print(f"Loaded model prediction: {prediction}")

    except Exception as e:
        logging.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()
