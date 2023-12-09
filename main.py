from chunking import process_text
from summarization import generate_summary
from transcribe import get_text
import logging


logging.basicConfig(filename='demo.log', encoding='utf-8', level=logging.DEBUG)

URL = "https://www.youtube.com/watch?v=tG-1XmWK8ms"
VIDEO_NAME = "andrew_huberman"

saved_to_file = get_text(url=URL, video_name=VIDEO_NAME)
logging.info(f"Saved to file: {saved_to_file}")
chunks = process_text(saved_to_file)
generate_summary(chunks, VIDEO_NAME)
