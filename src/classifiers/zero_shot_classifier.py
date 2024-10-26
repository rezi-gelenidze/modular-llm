import torch
from transformers import pipeline

from src import settings as local_settings


class ZeroShotClassifier:
    """
        Zero-shot classification using Hugging Face's pipeline

        Example usage:
        classifier = ZeroShotClassifier()
        result = classifier.classify("How to learn Python?")

        Returns:
        {
            'sequence': query,
            'labels': settings.CLASSIFICATION_LABELS,
            'scores': *vector of probability distribution over the labels
        }
    """
    def __init__(self):
        # Check if a GPU is available and set the device
        self.device = 0 if torch.cuda.is_available() else -1  # Use -1 for CPU
        print(f"Using device: {'GPU' if self.device == 0 else 'CPU'}")

        # Load the zero-shot classification pipeline with explicit model
        self.classifier = pipeline(
            "zero-shot-classification",
            model=local_settings.CLASSIFICATION_MODEL,
            device=self.device
        )

    def classify(self, query, candidate_labels=local_settings.CLASSIFICATION_LABELS):
        # Use the classifier to predict the topic
        return self.classifier(query, candidate_labels)


if __name__ == "__main__":
    # Test the classifier
    classifier = ZeroShotClassifier()

    # Get user input to classify
    query = input("Enter your query: ")

    # Classify the query
    result = classifier.classify(query)
    winner_label, winner_score = result["labels"][0], result["scores"][0]

    # Print the predicted topic and score
    print(
        f"Query: '{query}' | Predicted Topic: {winner_label}, Score: {winner_score:.2f}"
    )
