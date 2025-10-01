from io import BytesIO

import numpy as np
import soundfile as sf
import speech_recognition as sr
import whisper

model = whisper.load_model("base")

def recognize_whisper(audio: sr.AudioData, language="en"):
    # Ask SR to convert to exactly 16kHz, 16-bit PCM mono WAV
    wav_data = audio.get_wav_data(convert_rate=16000, convert_width=2)

    # Read as int16 so we control scaling precisely
    data, samplerate = sf.read(BytesIO(wav_data), dtype="int16", always_2d=False)

    # Downmix if stereo (should already be mono, but be defensive)
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Scale int16 -> float32 in [-1, 1]
    data = (data.astype(np.float32, copy=False) / 32768.0)

    # Now samplerate should be 16000; no librosa needed
    result = model.transcribe(data, language=language, fp16=False)
    return result["text"]
