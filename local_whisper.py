from io import BytesIO

import soundfile as sf
import speech_recognition as sr
import whisper

# Load Whisper model
model = whisper.load_model("base")

def recognize_whisper(audio: sr.AudioData, language="en"):
    # Convert SpeechRecognition AudioData to wav bytes
    wav_data = audio.get_wav_data()

    # Decode wav bytes -> float32 numpy array
    data, samplerate = sf.read(BytesIO(wav_data))
    
    # Resample if not 16kHz
    if samplerate != 16000:
        import librosa
        data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
    
    # Run Whisper
    result = model.transcribe(data, language=language)
    return result["text"]
