import whisper
from pytube import YouTube
from transformers import pipeline
import os
from typing import List
import logging

logging.basicConfig(filename='demo.log', encoding='utf-8', level=logging.ERROR)
OUTPUT_DIR = "output"


def download_audio_from_youtube(url: str, video_name: str) -> str:
    """Download the audio from a YouTube video and save it as an MP3 file."""
    video_url = YouTube(url)
    video = video_url.streams.filter(only_audio=True).first()
    filename = video_name + ".mp3"
    video.download(output_path=OUTPUT_DIR,filename=filename)
    return os.path.join(OUTPUT_DIR, filename)


def load_whisper_model(model_name: str = "large"):
    """Load the medium multilingual Whisper model."""
    print("available models: ", whisper.available_models()) #  ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large']
    return whisper.load_model(model_name)


def transcribe_audio_to_text(model, audio_path: str, language: str = "English"):
    """Transcribe the audio using the Whisper model."""
    return model.transcribe(audio_path, fp16=False, language=language)


def save_text_to_file(text: str, file_name: str):
    """Save the transcribed text to a file."""
    try:
        with open(file_name, "w+") as file:
            file.write(text)
    except (IOError, OSError, FileNotFoundError, PermissionError) as e:
        logging.debug(f"Error in file operation: {e}")


def get_text(url: str, video_name: str) -> str:
    model = load_whisper_model()
    audio_path = download_audio_from_youtube(url, video_name)
    result = transcribe_audio_to_text(model, audio_path)
    output_to = os.path.join(OUTPUT_DIR, video_name + ".txt")
    save_text_to_file(result["text"], output_to)
    return output_to
