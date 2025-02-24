import os
from flask import Flask, url_for, request, render_template
import librosa
import numpy as np
import pickle
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv

# Load the trained model
filename = "models/emotionmodel.pkl"
s = pickle.load(open(filename, 'rb'))

# Emotion-to-image mapping
d = {
    "fearful": "https://firebasestorage.googleapis.com/v0/b/myproject-d9de9.appspot.com/o/59-596556_fear-clipart-fear-emotion-cartoon-face-of-fear-removebg-preview.png?alt=media&token=c451d9cc-5fa7-47e1-bfa6-4be37e358ea8",
    "calm": "https://d29fhpw069ctt2.cloudfront.net/clipart/100203/preview/smiling_face_of_a_child_2_preview_9c89.png",
    "happy": "https://firebasestorage.googleapis.com/v0/b/myproject-d9de9.appspot.com/o/565-5650281_happy-boy-clipart-can-do-it-png-transparent-removebg-preview.png?alt=media&token=5964f656-e102-4f85-bb5c-ad4209209e39",
    "sad": "https://firebasestorage.googleapis.com/v0/b/myproject-d9de9.appspot.com/o/202-2022552_emotional-clipart-sad-dad-sad-clip-art-removebg.png?alt=media&token=3f1938a7-790e-4923-aea7-7f81ee2807b9",
    "angry": "https://firebasestorage.googleapis.com/v0/b/myproject-d9de9.appspot.com/o/clipart466731.png?alt=media&token=8dd82f61-b3ef-46f2-86c7-e1cd61f24ff3",
    "disgust": "https://firebasestorage.googleapis.com/v0/b/emotiondetection-1ccf6.appspot.com/o/photo_2022-01-02_20-00-18-removebg-preview.png?alt=media&token=38f24571-addd-439f-b47c-28928541876a"
}

# Function to extract audio features
def extract_feature(file_name, mfcc=True, chroma=True):
    try:
        X, sample_rate = librosa.load(file_name, sr=None)
        result = np.array([])
        if chroma:
            stft = np.abs(librosa.stft(X))
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        return result
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    result = False
    return render_template('inputfile.html', result1=result)

@app.route('/', methods=["POST"])
def home():
    try:
        result = True
        audio = request.files.get('shashifile')
        if not audio:
            raise ValueError("No audio file uploaded")
        
        # Save the uploaded file temporarily
        audio_path = "temp_audio.wav"
        audio.save(audio_path)
        
        feature = extract_feature(audio_path, mfcc=True, chroma=True)
        if feature is None:
            raise ValueError("Invalid audio format or error in feature extraction")
        
        p = s.predict([feature])
        r = [p[0], d[p[0]]]
        print(r)
        os.remove(audio_path)  # Clean up temporary file
        return render_template('inputfile.html', result1=result, r1=r)
    except Exception as e:
        print(f"Error: {e}")
        return render_template('inputfile.html', r2='Please enter a valid audio format')

@app.route('/record')
def record1():
    return render_template('record.html')

@app.route('/record', methods=['POST'])
def record():
    try:
        freq = 44100
        duration = 5
        recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
        sd.wait()
        audio_path = "recording1.wav"
        wv.write(audio_path, recording, freq, sampwidth=2)
        
        result = True
        feature = extract_feature(audio_path, mfcc=True, chroma=True)
        if feature is None:
            raise ValueError("Error in feature extraction")
        
        p = s.predict([feature])
        r = [p[0], d[p[0]]]
        print(r)
        return render_template('inputfile.html', result1=result, r1=r, a=audio_path)
    except Exception as e:
        print(f"Error: {e}")
        return render_template('inputfile.html', r2='Error during recording')

if __name__ == '__main__':
    app.run(debug=True)
