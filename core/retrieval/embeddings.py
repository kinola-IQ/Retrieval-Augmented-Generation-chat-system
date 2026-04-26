"""Handles Embedding generation"""
from typing import List
from xml.parsers.expat import model
import tiktoken
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter)

# custom modules
from ..utils.config import huggingface_config


# document spliter functions
def semantic_split(
    embedding_model: str = huggingface_config()[3],
    breakpoint_threshold_type: str = 'gradient',
    breakpoint_threshold_amount: float = 0.8
):
    """splits by semantic meaning"""
    # Initialize the SemanticChunker
    text_splitter = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount
        )
    return text_splitter


def recursive_split(
    separators: List[str] = ['\n\n', '\n', ' ', ''],
    chunk_size: int = 1000,
    chunk_overlap: int = 200):

    """splits by characters recursively"""
    text_splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
        )
    return text_splitter


def load_and_split_document(file_path: str, split_type: str = 'semantic_split'):
    """Loads a PDF document, splits it into chunks, \
        and generates embeddings for each chunk."""
    # Load the PDF document
    if file_path[-4:-1] != 'pdf':
        raise ValueError(
            f"Invalid file type: {file_path}. Only PDF files are supported."
            )
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    try:
        if split_type == 'semantic_split':
            doc_splitter = semantic_split()
        elif split_type == 'recursive_split':
            doc_splitter = recursive_split()
        chunks = doc_splitter.split_documents(documents)
        return chunks
    except Exception as err:
        raise ValueError(
                f"Invalid split type: {split_type}. \
                    Must be 'semantic_split' or 'recursive_split'.") from err


# query splitter functions
def character_split(separator: str = ' ', chunk_size: int = 24, chunk_overlap: int =3):
    """splits queries by characters"""
    text_splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter


def token_split(
    chunk_size: int = 10,
    chunk_overlap: int = 3,
    model_name: str = 'gpt-4o-mini'):
    """splits queries by tokens"""

    # ensuring token counts are accurate
    encoding = tiktoken.encoding_for_model(model_name)

    # creating splitter
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name=encoding.name
    )
    return text_splitter


def load_and_split_query(query: str, split_type: str = 'character_split'):
    """splits a query into chunks"""
    try:
        if split_type == 'character_split':
            text_splitter = character_split(separator=' ')
        elif split_type == 'token_split':
            text_splitter = token_split()
        chunks = text_splitter.split_text(query)
        return chunks

    except Exception as err:
        raise ValueError(
            f'input {err}: invalid input in split type'
        ) from err


def create_embeddings(model: str, huggingface_config()[3], chunks: List[float]):
    """creates embeddings from either queries or documents"""
    embedding_model = HuggingFaceEmbeddings(model_name=model)
    embeddings = embedding_model.embed_documents(chunks)
    return embeddings
