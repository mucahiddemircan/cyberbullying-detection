import os
import joblib
from pathlib import Path
from typing import List, Dict, Any
from .preprocessor import clean_text

class Predictor:
    """
    Service responsible for loading ML models and performing predictions.
    """
    def __init__(self):
        self.models_path = Path(__file__).parent.parent / "models"
        self.models: Dict[str, Any] = {}
        self.vectorizers: Dict[str, Any] = {}
        self.label_mapping = {
            0: "age",
            1: "ethnicity",
            2: "gender",
            3: "not_cyberbullying",
            4: "other_cyberbullying",
            5: "religion"
        }
        self.load_artifacts()

    def load_artifacts(self):
        """Loads all .pkl model and vectorizer files from the models directory."""
        if not self.models_path.exists():
            print(f"Warning: Models directory not found at {self.models_path}")
            return

        for file in os.listdir(self.models_path):
            name = file.replace("_model.pkl", "").replace("_vectorizer.pkl", "")
            if file.endswith("_model.pkl"):
                self.models[name] = joblib.load(self.models_path / file)
            elif file.endswith("_vectorizer.pkl"):
                self.vectorizers[name] = joblib.load(self.models_path / file)

    def get_available_models(self) -> List[str]:
        """Returns the names of all loaded models."""
        return list(self.models.keys())

    def predict(self, text: str, model_name: str) -> Dict[str, Any]:
        """
        Processes text and returns a prediction with confidence scores.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        # Determine which vectorizer to use based on the model name
        vectorizer_name = "TFIDF" if "TFIDF" in model_name else "BoW"
        vectorizer = self.vectorizers.get(vectorizer_name)
        
        if not vectorizer:
            raise ValueError(f"Vectorizer {vectorizer_name} not found")

        # Preprocess and vectorize the input text
        cleaned_text = clean_text(text)
        vectorized_text = vectorizer.transform([cleaned_text])
        
        # Perform prediction
        model = self.models[model_name]
        prediction = model.predict(vectorized_text)[0]
        
        prediction_index = int(prediction)
        label = self.label_mapping.get(prediction_index, "unknown")

        # Extract class probabilities if available
        probabilities = {}
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vectorized_text)[0]
            for i, prob in enumerate(probs):
                probabilities[self.label_mapping.get(i, f"class_{i}")] = float(prob)

        return {
            "prediction": label,
            "prediction_index": prediction_index,
            "probabilities": probabilities,
            "model_used": model_name,
            "vectorizer_used": vectorizer_name
        }

predictor = Predictor()


