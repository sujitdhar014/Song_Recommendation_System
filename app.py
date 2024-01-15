import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('model.h5')
import json
import random
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))


# music spotify
from spotipy.oauth2 import SpotifyClientCredentials

client_credentials_manager = SpotifyClientCredentials(client_id='9078ae3fbb0645fdb6ca17261df7e710', client_secret='0b5835cfa4094650b0e95f9616004cee')

from spotipy import Spotify

spotify = Spotify(client_credentials_manager=client_credentials_manager)

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res





# laoding models
df = pickle.load(open('df.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))

from flask import Flask, request, render_template
def recommendation(song_df):
    try:
        idx = df[df['song'] == song_df].index[0]
    except IndexError:
        # Handle the case when the song name is not found in the DataFrame
        return ["I'm sorry, I couldn't find that song."]

    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])

    songs = []
    for m_id in distances[1:9]:
        song_name = df.iloc[m_id[0]].song
        # Use the Spotify API to fetch song details
        results = spotify.search(q=f"track:{song_name}", type='track')
        if results and results['tracks']['items']:
            track_info = results['tracks']['items'][0]
            song_with_image = {
                'song_name': song_name,
                'image_url': track_info['album']['images'][0]['url'] if 'images' in track_info['album'] else None,
                'spotify_url': track_info['external_urls']['spotify'] if 'external_urls' in track_info else None
            }
            songs.append(song_with_image)
        else:
            songs.append({'song_name': song_name, 'image_url': None, 'spotify_url': None})

    return songs

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("/index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

@app.route('/', methods=['GET', 'POST'])
def index():
    names = list(df['song'].values)
    songs = []

    if request.method == 'POST':
        user_song = request.form['names']
        songs = recommendation(user_song)

    return render_template('index.html', name=names, songs=songs)
# save chat



if __name__ == "__main__":
    app.run(debug=True,port=5008)

