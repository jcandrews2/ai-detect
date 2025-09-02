from src.models.random_forest import RandomForest
from src.data.data_loader import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
import uvicorn
from src.api.api import app

def main():

    # Load and prepare the data
    print("===== Preparing data =====\n")
    # data_loader = DataLoader("~/Downloads/balanced_ai_human_prompts.csv")
    data_loader = DataLoader("~/Downloads/AI_Human.csv")
    data_loader.prepare()
    
    X_train, y_train = data_loader.get_train_data()
    X_test, y_test = data_loader.get_test_data()

    # Create and train the model
    vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 3),
            max_features=5000,
            min_df=2,
            max_df=0.95
        )
    model = RandomForest(vectorizer)

    model.train(X_train, y_train)
    print()
    print("===== Evaluating model =====\n")
    model.evaluate(X_test, y_test)
    model.save("src/models/saved", "rf_model.pkl")

    # Test loading and using the saved model
    print("\n===== Testing saved model =====")
    loaded_model = RandomForest.load("src/models/saved", "rf_model.pkl")
    
    # Verify it works with some example predictions
    test_text = "The rise of artificial intelligence has brought both innovation and uncertainty to the field of written communication. As large language models become increasingly sophisticated, the distinction between human and machine writing grows more difficult to identify. Traditional machine learning methods, such as logistic regression or support vector machines, can still provide strong baselines for classification when paired with textual features like TF-IDF and n-grams. However, as AI output improves in fluency and variety, the boundaries blur, forcing researchers to rely on more subtle signals such as perplexity, stylistic variation, and contextual coherence. Detecting AI-generated text is not simply a technical challenge—it is also a societal one, raising questions about authorship, authenticity, and trust in digital information. Ultimately, the task demands both computational rigor and careful consideration of how these tools are applied in real-world settings."

    pred, prob = loaded_model.predict(test_text)
    print(f"\nTest prediction (1=AI, 0=Human): {pred[0]}")
    print(f"Confidence: {prob[0].max():.2%}")

if __name__ == "__main__":
    main()

    # uvicorn.run("src.api.api:app", host="0.0.0.0", port=8000, reload=True)
    # source venv/bin/activate
    # python -m src.main