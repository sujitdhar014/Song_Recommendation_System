import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from keras.preprocessing import image
from keras.models import model_from_json
import numpy as np

nltk.download('vader_lexicon')

class MoodAnalyzer:
    def __init__(self):
        # Load the pre-trained model architecture and weights for emotion detection
        model_architecture = '/Users/sujit/Documents/IND3001Sujit/data.json'
        model_weights = '/Users/sujit/Documents/IND3001Sujit/model.h5'

        with open(model_architecture, 'r') as json_file:
            loaded_model_json = json_file.read()
            self.emotion_model = model_from_json(loaded_model_json)
            self.emotion_model.load_weights(model_weights)

        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        # Initialize sentiment analysis
        self.sid = SentimentIntensityAnalyzer()

    def analyze_emotion(self, image_path):
        # Load an image for emotion prediction
        img = image.load_img(image_path, target_size=(48, 48), grayscale=True)

        # Convert the image to a numpy array
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Make a prediction for emotion
        emotion_predictions = self.emotion_model.predict(img_array)

        # Get the predicted emotion label
        predicted_emotion = self.emotion_labels[np.argmax(emotion_predictions)]
        return predicted_emotion

    def analyze_sentiment(self, text):
        # Analyze sentiment using VADER
        sentiment_scores = self.sid.polarity_scores(text)
        compound_score = sentiment_scores['compound']

        if compound_score >= 0.05:
            return 'Positive'
        elif compound_score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

# Example of using the MoodAnalyzer in a chatbot
def chatbot():
    print("Chatbot: Hi there! How are you feeling today? You can also share an image for emotion detection.")
    
    mood_analyzer = MoodAnalyzer()

    while True:
        user_input = input("You: ")

        if user_input.lower() in ['bye', 'exit', 'quit']:
            print("Chatbot: Goodbye!")
            break

        if user_input.lower() == 'image':
            image_path = input("Enter the path to the image: ")
            mood = mood_analyzer.analyze_emotion(image_path)
            print(f"Chatbot: Based on the image, it seems like you're feeling {mood}.")
        else:
            mood = mood_analyzer.analyze_sentiment(user_input)
            print(f"Chatbot: It seems like you're feeling {mood}.")

if __name__ == "__main__":
    chatbot()
