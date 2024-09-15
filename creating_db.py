from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai
from dotenv import load_dotenv
import os
import shutil
import time

load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']


EMBEDDING_RPM = 3000  
EMBEDDING_RPD = 200   
REQUEST_DELAY = 60 / EMBEDDING_RPM  

CHROMA_PATH = "chroma"
DATA_PATH = "data"

MAX_RETRIES = 5
BACKOFF_FACTOR = 2


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings = OpenAIEmbeddings()

    try:
        db = Chroma.from_documents(
            chunks, embeddings, persist_directory=CHROMA_PATH
        )
        db.persist()
        print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
    except openai.error.OpenAIError as e:
        handle_rate_limit(chunks, embeddings)



def handle_rate_limit(chunks, embeddings):
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            print(f"Retrying ({retry_count + 1}/{MAX_RETRIES})...")
            db = Chroma.from_documents(
                chunks, embeddings, persist_directory=CHROMA_PATH
            )
            db.persist()
            print(f"Successfully saved after {retry_count + 1} retries.")
            break
        except openai.error.OpenAIError as e:
            wait_time = BACKOFF_FACTOR ** retry_count
            print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retry_count += 1

    if retry_count == MAX_RETRIES:
        print(f"Failed after {MAX_RETRIES} retries. Please try again later.")
        raise e


if __name__ == "__main__":
    main()
