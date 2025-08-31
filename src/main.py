from models.random_forest import RandomForest
from data.data import DataLoader

def main():
    # Prepare data
    data_loader = DataLoader("src/data/balanced_ai_human_prompts.csv")
    data_loader.prepare()

    # Create and train model
    model = RandomForest(data_loader)
    model.train()
    model.evaluate()

    # Try some predictions
    print("\nTesting predictions:")
    ai_like = "Learning new skills can feel overwhelming at first, but persistence often leads to surprising progress. Even when mistakes happen, they create opportunities to understand concepts more deeply. For example, many programmers struggle with the basics of machine learning, yet small steps like experimenting with vectorizers and classifiers build confidence. With consistent practice, what once felt confusing begins to make sense. Ultimately, growth comes not from avoiding challenges but from working through them patiently."
    human_like = "Unfortunately, legalizing Cocaine in the U.S. is a much larger undertaking than you might think. Although cocaine has similar levels of addictiveness to alcohol, with a reported 17% of all users becoming dependent on cocaine and 15% of all users becoming dependent on alcohol, cocaine is seen by many Americans as a highly addictive and dangerous substance to be circulating in the public. "
    
    pred_ai, prob_ai = model.predict(ai_like)
    pred_human, prob_human = model.predict(human_like)
    
    print(f"\nAI-like text:")
    print(f"Prediction (1=AI, 0=Human): {pred_ai[0]}")
    print(f"Confidence: {prob_ai[0].max():.2%}")
    
    print(f"\nHuman-like text:")
    print(f"Prediction (1=AI, 0=Human): {pred_human[0]}")
    print(f"Confidence: {prob_human[0].max():.2%}")

if __name__ == "__main__":
    main()