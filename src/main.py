from models.random_forest import RandomForest
from data.data_loader import DataLoader

def main():
    # Load and prepare the data
    print("===== Preparing data =====\n")
    data_loader = DataLoader("~/Downloads/AI_Human.csv")
    data_loader.prepare()
    X_train, y_train = data_loader.get_train_data()
    X_test, y_test = data_loader.get_test_data()

    # Create and train the model
    print("===== Training model =====\n")
    model = RandomForest()
    model.train(X_train, y_train)
    model.evaluate(X_test, y_test)
    model.save("saved", "rf_model.pkl")

    # Test loading and using the saved model
    print("\n===== Testing saved model =====")
    loaded_model = RandomForest.load("saved/rf_model.pkl")
    
    # Verify it works with some example predictions
    test_text = "This is a test sentence to verify the loaded model works."
    pred, prob = loaded_model.predict(test_text)
    print(f"\nTest prediction (1=AI, 0=Human): {pred[0]}")
    print(f"Confidence: {prob[0].max():.2%}")

if __name__ == "__main__":
    main()