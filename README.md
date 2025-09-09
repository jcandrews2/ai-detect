# AI Text Detection API

An AI/ML API that predicts whether text is human-written or AI-generated. The API offers multiple classification models (Logistic Regression, Random Forest, Neural Networks) paired with different text encoding methods (GloVe word embeddings or TF-IDF vectorization). All of the models are trained on a Kaggle dataset of human and AI-generated essays.

🌐 **Live API**: [http://ec2-3-131-93-133.us-east-2.compute.amazonaws.com/api/v1/](http://ec2-3-131-93-133.us-east-2.compute.amazonaws.com/api/v1/)

## ✨ Features

- **Classification Models**:
- Logistic Regression
- Random Forest
- Neural Network

- **Text Encoding**:
- GloVe (Global Vectors for Word Representation)
- TF-IDF (Term Frequency-Inverse Document Frequency)

## 🛠️ Technology Stack

- **Programming Language**:
  - Python

- **Web Framework**:
  - FastAPI (REST API)

- **Machine Learning**:
  - PyTorch (Neural Networks)
  - scikit-learn (Traditional ML)

- **Data Processing**:
  - Pandas (Data manipulation)

- **Deployment**:
  - AWS EC2 (Cloud hosting)

- **Testing & Documentation**:
  - Postman (API testing)

## 📋 API Documentation

### 🔌 Endpoints

#### 1. Classification Endpoint
```http
POST /api/v1/classify
```

**Request Body**:
```json
{
"text": "Text to classify",
"encoder": "glove",  // Options: "glove", "tfidf"
"model": "logistic_regression"  // Options: "logistic_regression", "random_forest", "neural_network"
}
```

**Response**:
```json
{
"model": "Logistic Regression",
"encoder": "glove",
"prediction": "Human",
"confidence": {
    "human": "85.50%",
    "ai": "14.50%"
  }
}
```

#### 2. Info Endpoint
```http
GET /api/v1/info
```
Returns model information and usage examples.

## ⚡ Getting Started

### 📋 Prerequisites
- Python 3.12+
- Pip package manager

### 💻 Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/ai-detect.git
cd ai-detect
```

2. Create and activate virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download and prepare the dataset
- Visit [Kaggle AI Vs Human Text Dataset](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text)
- Download and place the CSV file in `~/Downloads`

5. Train the models
- Open `src/main.py`
- Uncomment the `main()` function call at the bottom of the file
- Run the training script:
```bash
python -m src.main
```
- This will train all models and save them in `src/models/saved/`

6. Start the API server
- Open `src/api/api.py`
- Uncomment the uvicorn line at the bottom of the file
- Run the server:
```bash
python -m src.main
```

The API will be available at `http://localhost:8000`

## 📁 Project Structure

```
ai-detect/
├── src/
│   ├── api/           # FastAPI application
│   ├── models/        # ML model implementations
│   ├── encoders/      # Text encoding modules
│   └── data/          # Data processing utilities
├── tests/             # Unit tests
└── requirements.txt   # Project dependencies
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

Jimmy Andrews
- Email: jcandrews2@icloud.com

## 📚 Acknowledgments

- Trained using the Kaggle dataset [AI Vs Human Text](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text)
