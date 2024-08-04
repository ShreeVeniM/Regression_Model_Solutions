from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle

def train_linear_regression(x_train, y_train):
    try:
        model = LinearRegression().fit(x_train, y_train)
        return model
    except Exception as e:
        print(f"Error training Linear Regression model: {e}")
        return None

def train_decision_tree(x_train, y_train, max_depth=3, max_features=10, random_state=567):
    try:
        model = DecisionTreeRegressor(max_depth=max_depth, max_features=max_features, random_state=random_state).fit(x_train, y_train)
        return model
    except Exception as e:
        print(f"Error training Decision Tree model: {e}")
        return None

def train_random_forest(x_train, y_train, n_estimators=200, criterion='absolute_error'):
    try:
        model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion).fit(x_train, y_train)
        return model
    except Exception as e:
        print(f"Error training Random Forest model: {e}")
        return None

def save_model(model, filename):
    try:
        pickle.dump(model, open(filename, 'wb'))
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model(filename):
    try:
        model = pickle.load(open(filename, 'rb'))
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
