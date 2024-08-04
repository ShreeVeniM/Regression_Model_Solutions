import os

from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn import tree

def evaluate_model(model, x_train, y_train, x_test, y_test):
    try:
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
        train_mae = mean_absolute_error(train_pred, y_train)
        test_mae = mean_absolute_error(test_pred, y_test)
        return train_mae, test_mae
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return None, None

def plot_tree(model, filename='tree.png'):
    try:
        # Create 'assets' folder if it doesn't exist
        os.makedirs('assets', exist_ok=True)
        
        # Define the path to save the file
        file_path = os.path.join('assets', filename)
        
        plt.figure(figsize=(20,10))
        tree.plot_tree(model, feature_names=model.feature_names_in_, filled=True)
        plt.savefig(filename, dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error plotting tree: {e}")
