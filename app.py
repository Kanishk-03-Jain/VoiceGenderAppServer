from flask import Flask, request, jsonify, send_file
import os
import base64
from flask_cors import CORS
from pydub import AudioSegment
import requests
from utils import create_model, extract_feature, is_valid_wav, transcribe_audio, verify_transcription
import io
import torch
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, pipeline
from diffusers import StableDiffusionXLPipeline
from dotenv import load_dotenv
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

TEXT = "Hi this is an example text for getting similarity score."
app = Flask(__name__)
CORS(app)
model = create_model()
model.load_weights("models/model.h5")

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print("Using:", device, torch_dtype)

# Florence-2 model and processor
florence_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large", 
    torch_dtype=torch_dtype, 
    trust_remote_code=True
).to(device)

florence_processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-large", 
    trust_remote_code=True
)

# Summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

# Stable Diffusion XL model
sd_pipe = StableDiffusionXLPipeline.from_pretrained(
    "segmind/Segmind-Vega", 
    torch_dtype=torch.float16, 
    use_safetensors=True, 
    variant="fp16"
)
sd_pipe.to(device)

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

@app.route("/generate_avatar", methods=["POST"])
def generate_image():
    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"error": "Missing 'image' in request body"}), 400

        base64_image = data["image"]

        # Remove prefix if present (e.g., "data:image/png;base64,")
        if "," in base64_image:
            base64_image = base64_image.split(",")[1]

        image_bytes = base64.b64decode(base64_image)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        task_prompt = "<MORE_DETAILED_CAPTION>"
        inputs = florence_processor(text=task_prompt, images=image, return_tensors="pt").to(device, torch_dtype)
        generated_ids = florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
        generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_caption = florence_processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(image.width, image.height)
        )
        
        detailed_caption = parsed_caption["<MORE_DETAILED_CAPTION>"]
        prompt_text = (
            "3D cartoon-style portrait of a " + detailed_caption[35:] + 
            " Ghibli-style realism, hyper-detailed eyes, soft textures, cinematic depth of field, beautifully rendered in high resolution."
        )

        short_caption = summarizer(prompt_text, max_length=80, min_length=76, do_sample=False)[0]['summary_text']
        print("Short caption:", short_caption)

        gen_image = sd_pipe(
            prompt=short_caption,
            generator=torch.manual_seed(1),
            num_inference_steps=40,
        ).images[0]

        # Return image as response
        img_io = BytesIO()
        gen_image.save(img_io, 'PNG')
        img_io.seek(0)
        print("Image generated successfully.")
        encoded_img = base64.b64encode(img_io.getvalue()).decode('utf-8')

        return jsonify({"generated_image": encoded_img})
        # return send_file(img_io, mimetype='image/png')
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__=="__main__":
    app.run(host="0.0.0.0", port=7000, debug=True)
