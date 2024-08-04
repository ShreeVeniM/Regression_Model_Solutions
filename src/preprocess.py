import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df, target_variable, test_size=0.2, random_state=1234):
    try:
        # Debug: Print the columns in the DataFrame
        print("Columns in DataFrame:", df.columns)

        x = df.drop(target_variable, axis=1)
        y = df[target_variable]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        return x_train, x_test, y_train, y_test
    except Exception as e:
        print(f"Error splitting data: {e}")
        return None, None, None, None