from models.math_model import MathModel
from models.coding_model import CodingModel
from models.life_advice_model import LifeAdviceModel
from models.general_model import GeneralModel

from src.classifiers.zero_shot_classifier import ZeroShotClassifier
from src import settings as local_settings

math_model = MathModel()
coding_model = CodingModel()
life_advice_model = LifeAdviceModel()
general_model = GeneralModel()
classifier = ZeroShotClassifier()


def handle_query(query):
    """ Classifies query and finds answer in a specific model """
    result = classifier.classify(query)
    winner_label, confidence = result["labels"][0], result["scores"][0]

    print(f"Using model: {winner_label}, Confidence: {confidence:.2f}")

    # check threshold to decide if using topic model or general model
    if confidence > local_settings.CLASSIFICATION_THRESHOLD:
        pass
        # answer = general_model.answer(query)
    else:
        if winner_label == 'math':
            pass
            # answer = math_model.answer(query, "Math context")
        elif winner_label == 'coding':
            pass
            # answer = coding_model.answer(query, "Coding context")
        elif winner_label == 'life advice':
            pass
            # answer = life_advice_model.answer(query, "Life advice context")
        else:
            pass
            # answer = general_model.answer(query)

    return ""


if __name__ == "__main__":
    user_query = input("Enter your query: ")
    response = handle_query(user_query)
    print("Response:", response)
