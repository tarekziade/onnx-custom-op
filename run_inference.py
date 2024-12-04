import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import os

VERSION = "v0.1.1"


class IntentClassifier:
    def __init__(self):
        self.id2label = {
            0: "information_intent",
            1: "yelp_intent",
            2: "navigation_intent",
            3: "travel_intent",
            4: "purchase_intent",
            5: "weather_intent",
            6: "translation_intent",
            7: "unknown",
        }
        self.label2id = {label: id for id, label in self.id2label.items()}
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Mozilla/mobilebert-uncased-finetuned-LoRA-intent-classifier"
        )
        model_path = "model_quantized.firefox.onnx"
        self.ort_session = ort.InferenceSession(model_path)

    def find_intent(self, sequence, verbose=False):
        inputs = self.tokenizer(
            sequence,
            return_tensors="np",  # ONNX requires inputs in NumPy format
            padding="max_length",  # Pad to max length
            truncation=True,  # Truncate if the text is too long
            max_length=64,
        )

        # Convert inputs to NumPy arrays
        onnx_inputs = {k: v for k, v in inputs.items()}

        # Run the ONNX model
        logits = self.ort_session.run(None, onnx_inputs)[0]

        # Get the prediction
        prediction = np.argmax(logits, axis=1)[0]
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        rounded_probabilities = np.round(probabilities, decimals=3)

        pred_result = self.id2label[prediction]
        proba_result = dict(
            zip(self.label2id.keys(), rounded_probabilities[0].tolist())
        )

        if verbose:
            print(sequence + " -> " + pred_result)
            print(proba_result, "\n")

        return pred_result, proba_result


def main():
    text_list = [
        "floor repair cost",
        "pet store near me",
        "who is the us president",
        "italian food",
        "sandwiches for lunch",
        "cheese burger cost",
        "What is the weather today",
        "what is the capital of usa",
        "cruise trip to carribean",
    ]
    cls = IntentClassifier()
    for sequence in text_list:
        cls.find_intent(sequence, verbose=True)


if __name__ == "__main__":
    main()
