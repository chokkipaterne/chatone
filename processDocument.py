#!/usr/bin/env python3
import os
import glob
from typing import List
from dotenv import load_dotenv
from multiprocessing import Pool
from tqdm import tqdm
import streamlit as st

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.docstore.document import Document
from chromadb.config import Settings
from constants import llms, openai_index, chunk_size, chunk_overlap


load_dotenv()

# Load environment variables
#embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
#chunk_size = 500
#chunk_overlap = 50

# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    #".csv": (CSVLoader, {"csv_args": {"delimiter": ";"}}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]
    results = []
    docs = load_single_document(filtered_files[0])
    results.extend(docs)

    """with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()"""

    return results

def process_documents(project_code:str, ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    source_directory = os.environ.get('SOURCE_DOCS', 'source_docs') + "/" +  project_code
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts

def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False

def store_document(project_code, llm_name, embeddings_model_name, main_model, api_key):
    persist_directory = os.environ.get('SOURCE_DBS', 'source_dbs') + "/" +  project_code

    # Create embeddings
    if llm_name == llms[openai_index]:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key, model=embeddings_model_name)
    else:
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    # Define the Chroma settings
    CHROMA_SETTINGS = Settings(
            chroma_db_impl='duckdb+parquet',
            persist_directory=persist_directory,
            anonymized_telemetry=False
    )

    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        collection = db.get()
        texts = process_documents(project_code, [metadata['source'] for metadata in collection['metadatas']])
        print(f"Creating embeddings. May take some minutes...")
        db.add_documents(texts)
    else:
        os.makedirs(persist_directory)
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        texts = process_documents(project_code)
        print(f"Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None
    ss = st.session_state
    ss['list_sources'] =  get_list_sources(ss.get('project_code'), ss.get('llm'), ss.get('embeddings_model_name'), ss.get('api_key'))

    print(f"Ingestion complete! You can now query your documents")

def get_list_sources(project_code, llm_name, embeddings_model_name, api_key):
    persist_directory = os.environ.get('SOURCE_DBS', 'source_dbs') + "/" +  project_code

    # Create embeddings
    if llm_name == llms[openai_index]:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key, model=embeddings_model_name)
    else:
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    # Define the Chroma settings
    CHROMA_SETTINGS = Settings(
            chroma_db_impl='duckdb+parquet',
            persist_directory=persist_directory,
            anonymized_telemetry=False
    )
    list_sources = []

    if does_vectorstore_exist(persist_directory):
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        collection = db.get()
        for metadata in collection['metadatas']:
            if metadata['source'] not in list_sources:
                list_sources.append(metadata['source'])

        #print(f"list of documents.......")
        #print(list_sources)
    return list_sources
   
