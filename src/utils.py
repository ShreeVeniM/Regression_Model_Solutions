def print_evaluation_results(train_mae, test_mae):
    try:
        print(f"Train Mean Absolute Error: {train_mae}")
        print(f"Test Mean Absolute Error: {test_mae}")
    except Exception as e:
        print(f"Error printing evaluation results: {e}")
