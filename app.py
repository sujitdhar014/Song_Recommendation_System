import os
import random
import pickle
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, render_template
from keras.models import load_model
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials

nltk.download('popular')

# NLP setup
lemmatizer = WordNetLemmatizer()

# Load chat model
model = load_model('model.h5')
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

# Load recommendation components
df = pickle.load(open('df.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
song_name_to_index = pickle.load(open('song_index.pkl', 'rb'))

# Spotify setup
client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
spotify = Spotify(client_credentials_manager=client_credentials_manager)

# NLP functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: 
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I'm not sure how to help with that."

def chatbot_response(msg):
    ints = predict_class(msg, model)
    return getResponse(ints, intents)

# Song recomendation function >>>>>>

from difflib import get_close_matches

# Normalization function
def normalize(text):
    return text.lower().strip()

# Preprocess your mapping dictionary to use normalized keys
song_name_to_index = {normalize(k): v for k, v in song_name_to_index.items()}

def recommendation(song_name):
    song_name = normalize(song_name)

    # Fuzzy matching fallback if song not found
    if song_name not in song_name_to_index:
        close_matches = get_close_matches(song_name, song_name_to_index.keys(), n=1, cutoff=0.7)
        if close_matches:
            song_name = close_matches[0]
        else:
            return [{"song_name": "Song not found", "image_url": None, "spotify_url": None}]

    idx = song_name_to_index[song_name]

    # Safety check for similarity matrix bounds
    if idx >= similarity.shape[0]:
        print(f"Index {idx} is out of bounds for similarity matrix of shape {similarity.shape}")
        return [{"song_name": "Internal Error: Similarity index out of bounds", "image_url": None, "spotify_url": None}]

    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])

    songs = []
    for m_id in distances[1:10]:
        try:
            recommended_song_name = df.iloc[m_id[0]]['SongName']
            results = spotify.search(q=f"track:{recommended_song_name}", type='track', limit=1)
            if results['tracks']['items']:
                track = results['tracks']['items'][0]
                songs.append({
                    'song_name': recommended_song_name,
                    'image_url': track['album']['images'][0]['url'],
                    'spotify_url': track['external_urls']['spotify']
                })
            else:
                songs.append({'song_name': recommended_song_name, 'image_url': None, 'spotify_url': None})
        except Exception as e:
            print(f"Error fetching song: {recommended_song_name}, {e}")
            songs.append({'song_name': recommended_song_name, 'image_url': None, 'spotify_url': None})

    return songs



# Flask app setup
app = Flask(__name__)
app.static_folder = 'static'

@app.route("/", methods=["GET", "POST"])
def home():
    names = list(df['SongName'].values)
    songs = []

    if request.method == 'POST':
        user_song = request.form['names']
        songs = recommendation(user_song)

    return render_template("index.html", name=names, songs=songs)


@app.route("/get", methods=['GET'])
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

@app.route('/recommend', methods=['POST'])
def index():
    names = list(df['SongName'].values)
    songs = []

    if request.method == 'POST':
        user_song = request.form['names']
        songs = recommendation(user_song)

    return render_template('index.html', name=names, songs=songs)
# Assuming you're using Flask
@app.route('/songs')
def songs_page():
    songs = [
        {"song_name": "Song One", "image_url": "https://example.com/song1.jpg"},
        {"song_name": "Song Two", "image_url": ""},
        {"song_name": "Song Three", "image_url": "https://example.com/song3.jpg"}
    ]
    return render_template('test_design.html', songs=songs)

if __name__ == "__main__":
    app.run(debug=True, port=5008)
