from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

cache_dir = "/media/arko/work/.hfcache"
pdf_abs_location = "/home/arko/PycharmProjects/video-transcription/sample-data/rp.pdf"

loder = PyPDFLoader(pdf_abs_location)
documents: List[Document] = loder.load()
print(type(documents))
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=100,
)
texts: [Document] = text_splitter.split_documents(documents)

print(len(texts))

url = "http://localhost:6333"
collection_name = "test_db"


def get_chunk(position):
    return texts[position].page_content






