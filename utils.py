import numpy as np
import os
import librosa
import wave
import difflib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import assemblyai as aai
# from dotenv import load_dotenv
# load_dotenv()
aai.settings.api_key = os.environ.get("ASSEMBLYAI_API_KEY")

def transcribe_audio(wav_buffer):
    """Transcribe audio from BytesIO using AssemblyAI API"""
    wav_buffer.seek(0)  # Ensure the buffer is at the beginning
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(wav_buffer)
    return transcript.text

def verify_transcription(user_transcription, original_text):
    """Get the similarity between the user transcription and the original text"""
    if not user_transcription or not original_text:
        return 0.0
    # Normalize both texts by stripping whitespace and converting to lowercase
    similarity = difflib.SequenceMatcher(None, user_transcription.strip().lower(), original_text.strip().lower()).ratio()
    return round(similarity, 2)

def create_model(vector_length=128):
    """5 hidden dense layers from 256 units to 64, not the best model, but not bad."""
    model = Sequential()
    model.add(Dense(256, input_shape=(vector_length,)))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    # one output neuron with sigmoid activation function, 0 means female, 1 means male
    model.add(Dense(1, activation="sigmoid"))
    # using binary crossentropy as it's male/female classification (binary)
    model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")
    return model

def extract_feature(file, **kwargs):
    """
    Extract feature from audio file `file`
        Features supported:
            - MFCC (mfcc) (Mel Frequency Cepstral Coefficients) {short-term power spectrum of a sound}
            - Chroma (chroma) {12 different pitch classes}
            - MEL Spectrogram Frequency (mel) {frequencies of audio over time, but scaled according to the Mel scale}
            - Contrast (contrast) {difference between peaks and valleys}
            - Tonnetz (tonnetz) {tonal centroid of the audio — capturing harmonic relationships like chord structures, key, and mode.}
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    X, sample_rate = librosa.load(file, sr=None)
    if chroma or contrast:
        stft = np.abs(librosa.stft(X))  # STFT (Short-Time Fourier Transform) breaks the audio signal into short overlapping windows and computes the Fourier Transform (frequency spectrum) for each window.
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        result = np.hstack((result, tonnetz))
    return result

def is_valid_wav(file_path):
    """Check if the file is a valid 16-bit WAV file (mono/stereo)"""
    try:
        with wave.open(file_path, 'rb') as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            return channels in [1, 2] and sample_width == 2
    except Exception:
        return False
