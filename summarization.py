import os

from transformers import pipeline
from typing import Callable, List, Dict
import logging

from chunking import split_text_into_chunks
from transcribe import OUTPUT_DIR

logging.basicConfig(filename='demo.log', encoding='utf-8', level=logging.DEBUG)

bart_params = {
    "max_length": 124,
    "min_length": 30,
    "do_sample": False,
    "truncation": True,
    "repetition_penalty": 2.0,
}


def create_summarizer(model: str) -> Callable:
    summarizer = pipeline("summarization", model=model)
    return summarizer


def get_summary_bart(
    list_chunks: List[str], summarizer: Callable, summarization_params: Dict[str, int]
) -> str:
    # Generate summaries for each text chunk
    try:
        summaries = [
            summarizer(chunk, **summarization_params)[0]["summary_text"]
            for chunk in list_chunks
        ]
        return " ".join(summaries)
    except Exception as e:
        logging.error(f"Error generating summaries: {e}")
        return ""


def save_summary_to_file(summary: str, file_name: str) -> None:
    try:
        # Save the summary to a file
        with open(f"{file_name}", "a") as fp:
            fp.write(summary)
    except Exception as e:
        logging.error(f"Error saving summary to file: {e}")


def generate_summary(text_chunks: List[str], video_name: str) -> None:
    summarizer = create_summarizer("facebook/bart-large-cnn")
    summary = get_summary_bart(text_chunks, summarizer, bart_params)
    print(summary)
    summary_file_path = os.path.join(OUTPUT_DIR, f"summary_{video_name}" + ".txt")
    save_summary_to_file(summary, summary_file_path)
    if len(summary) > 5000:
        # If the summary is too long we can reapply the summarization function
        text_chunks = split_text_into_chunks(summary, max_tokens=1000)
        short_summary = get_summary_bart(text_chunks, summarizer, bart_params)
        short_summary_file_path = os.path.join(OUTPUT_DIR, f"short_summary_{video_name}" + ".txt")
        save_summary_to_file(short_summary, short_summary_file_path)
        logging.info("Summary saved to file.")
    else:
        logging.error("Error: Unable to generate summary.")