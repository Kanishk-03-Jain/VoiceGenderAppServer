from flask import Flask, request, jsonify
from flask_cors import CORS
from pydub import AudioSegment
import requests
from utils import create_model, extract_feature, is_valid_wav, transcribe_audio, verify_transcription
import io

TEXT = "Hi this is an example text for getting similarity score."
app = Flask(__name__)
CORS(app)
model = create_model()
model.load_weights("models/model.h5")


#Home page
@app.route("/", methods=["GET"])
def home():
    return "🎙️ Gender Prediction API is up. Use POST /predict to analyze voice."

@app.route("/text", methods=["GET"])
def display_text():
    return jsonify({"text": TEXT}), 200

@app.route("/transcribe", methods=["POST"])
def transcribe_and_verify():
    try:
        data = request.get_json()
        print(data)

        if not data or "url" not in data:
            return jsonify({'error': 'No URL provided'}), 400

        response = requests.get(data["url"])
        if response.status_code != 200:
            return jsonify({'error': 'Failed to download audio from URL'}), 400

        audio_bytes = io.BytesIO(response.content)
        audio = AudioSegment.from_file(audio_bytes)
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)

        transctiption = transcribe_audio(wav_buffer)
        if transctiption is None:
            return jsonify({'error': 'Failed to transcribe audio.'}), 400
        
        return jsonify({
            'transcription': transctiption,
            'similarity': verify_transcription(transctiption, TEXT)
        })

    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

#/predict post request
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print(data)

        if not data or "url" not in data:
            return jsonify({'error': 'No URL provided'}), 400

        response = requests.get(data["url"])
        if response.status_code != 200:
            return jsonify({'error': 'Failed to download audio from URL'}), 400

        audio_bytes = io.BytesIO(response.content)
        audio = AudioSegment.from_file(audio_bytes)
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)

        features = extract_feature(wav_buffer, mel=True).reshape(1, -1)
        if features is None:
            return jsonify({'error': 'Failed to extract features from audio.'}), 400

        # Making prediction
        prediction = model.predict(features, verbose=0)
        male_prob = prediction[0][0]
        female_prob = 1 - male_prob
        gender = "male" if male_prob > female_prob else "female"
        confidence = float(max(male_prob, female_prob))

        return jsonify({
            'gender': gender,
            'confidence': round(confidence, 4)
        })

    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
