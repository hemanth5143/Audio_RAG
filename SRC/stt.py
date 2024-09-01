import pyaudio
import wave
import openai
import numpy as np
import os

from dotenv import load_dotenv
load_dotenv()

# Function to record audio locally
def record_audio(filename, duration=5, sample_rate=44100):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording...")

    frames = []

    for i in range(0, int(sample_rate / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()


# Use OpenAI's Whisper API for transcription
def transcribe_audio(file_path):
    openai.api_key = os.getenv("OPEN_AI_API_KEY") # Replace with your OpenAI API key
    audio_file = open(file_path, "rb")
    transcription = openai.Audio.transcribe(
        model="whisper-1",
        file=audio_file
    )
    return transcription['text']

# Record audio
filename = "recorded_audio.wav"
record_audio(filename)

# Transcribe the recorded audio
transcription_text = transcribe_audio(filename)
print("Transcription:", transcription_text)