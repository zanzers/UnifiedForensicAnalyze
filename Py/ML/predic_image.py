import sys, json, os
import numpy as np
import pickle
from RandomForest import RandomForest



class RFClassifier:

    LABELS = {
        0: "Original",
        1: "Tampered",
        2: "AI-generated"

    }

    def __init__(self, model_path="rf_model.pkl"):
        self.model_path = os.path.join(os.path.dirname(__file__), model_path)
        self.model = self._load_model()

    def _load_model(self):
        with open(self.model_path, "rb") as f:
            model = pickle.load(f)
        return model

    def _load_json(self, json_input):
        if isinstance(json_input, str):
            with open(json_input, "r") as f:
                data = json.load(f)
        else:
            data = json_input

        if isinstance(data, list):
            data = data[0]


        if not isinstance(data, dict):
            raise ValueError("Invalid JSON structure. Must be a dict or list of dicts.")
        features = [float(v) for v in data.values() if isinstance(v, (int, float))]
        return features

    

    def predict(self, features):
      
        features = np.array(features).reshape(1, -1)
        all_preds = np.array([tree.predict(features)[0] for tree in self.model.trees])

        unique, counts = np.unique(all_preds, return_counts=True)
        votes = dict(zip(unique, counts))
        total_votes = len(all_preds)

        confidence_scores = {self.LABELS[k]: (v / total_votes) * 100 for k, v in votes.items()}

        final_pred = max(votes, key=votes.get)

    
        for lbl in self.LABELS.values():
            if lbl not in confidence_scores:
                confidence_scores[lbl] = 0.0

        return self.LABELS[final_pred], confidence_scores

    def predict_json(self, json_input):
        features = self._load_json(json_input)
        return self.predict(features)

            


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(json.dumps({"error": "Missing JSON path argument"}))
        sys.exit(1)

    
    json_path = sys.argv[1]
    rfclass = RFClassifier(model_path="rf_model.pkl")
    
    


    try:
        label, conf = rfclass.predict_json(json_path)

        result = {
            "RF_label": label,
            "RF_confidence": conf[label],
            "RF_probabilities": conf
        }
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
