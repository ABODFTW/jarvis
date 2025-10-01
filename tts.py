import logging
import time

import pyttsx3


def pyttsx3_engine(text: str) -> None:
    try:
        engine = pyttsx3.init()
        for voice in engine.getProperty("voices"):
            if "jamie" in voice.name.lower():
                engine.setProperty("voice", voice.id)
                break
        engine.setProperty("rate", 180)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
        time.sleep(0.3)
    except Exception as e:
        logging.error(f"❌ TTS failed: {e}")


# TTS setup
def speak_text(text: str, engine: str = "pyttsx3") -> None:
    if engine == "pyttsx3":
        pyttsx3_engine(text)


if __name__ == "__main__":
    speak_text("Hello, this is a test of the text to speech system.")
