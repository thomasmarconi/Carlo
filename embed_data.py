# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv

import os
import shutil

CHROMA_PATH = "chroma"

load_dotenv()

def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    DATA_PATH = "data/"
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=300,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents) # figure out how to set the meta data so we can set the title.
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    
    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # Can change this model if we find a better one. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    # embeddings = MistralAIEmbeddings(api_key=os.environ['MISTRAL_API_KEY']) # Idk why, this just wouldnt work
    Chroma.from_documents(
        chunks, embedding=embeddings, persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
