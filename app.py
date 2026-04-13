from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# # ─── Custom model class (MUST be defined before pickle.load) ──────────────────
# class MyLogisticRegression:
#     def __init__(self, lr=0.01, iterations=1000):
#         self.lr = lr
#         self.iterations = iterations
#         self.W = None
#         self.b = None

#     def _sigmoid(self, z):
#         return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # clip to avoid overflow

#     def predict(self, X):
#         linear = X.dot(self.W) + self.b
#         y_pred = self._sigmoid(linear)
#         return (y_pred >= 0.5).astype(int), y_pred

# ─── NLTK downloads ───────────────────────────────────────────────────────────
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ─── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # allow requests from Chrome extension

# ─── Load model + vectorizer (flexible paths) ────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_pickle(filename):
    """Try loading from current dir, then from a 'backend_fake_reviews_detection' subdir."""
    paths = [
        os.path.join(BASE_DIR, filename),
        os.path.join(BASE_DIR, 'backend_fake_reviews_detection', filename),
    ]
    for path in paths:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
    raise FileNotFoundError(f"Could not find {filename} in {paths}")

model      = load_pickle('logistic_model.pkl')
vectorizer = load_pickle('tfidf_vec.pkl')
stop_words = set(stopwords.words('english'))

# ─── Text preprocessing ───────────────────────────────────────────────────────
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    try:
        words = word_tokenize(text)
    except Exception:
        words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 1]
    return " ".join(words)

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return "🚀 Fake Review Detector API is running."

@app.route('/analyze_reviews', methods=['POST'])
def analyze_reviews():
    data = request.get_json(force=True, silent=True)

    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    reviews = data.get("reviews", [])

    if not reviews or not isinstance(reviews, list):
        return jsonify({"error": "No reviews provided"}), 400

    # Preprocess
    processed = [preprocess_text(r) for r in reviews]

    # Vectorize
    X = vectorizer.transform(processed)

    # Predict
    preds, probs = model.predict(X)

    total = len(reviews)
    per_review = preds.tolist() if hasattr(preds, 'tolist') else list(preds)
    fake  = int(sum(1 for p in per_review if p == 1))
    real  = total - fake

    return jsonify({
        "total":      total,
        "fake":       fake,
        "real":       real,
        "per_review": per_review,                              # 1 = fake, 0 = real per review
        "confidence": [round(float(p), 3) for p in probs],    # probability scores
    })

# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(" Model and vectorizer loaded successfully.")
    print(" Starting Flask server at http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)
