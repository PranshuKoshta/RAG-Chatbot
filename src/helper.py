from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document

from huggingface_hub import InferenceClient
from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings  # LOCAL MODEL
from dotenv import load_dotenv
import os


# Extract text from PDF files
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs

#Split the Data into Text Chunks
def text_split(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    text_chunk = text_splitter.split_documents(documents)
    return text_chunk


HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

load_dotenv()

# Local Embeddings 
def load_local_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name=HF_MODEL)
    return embeddings

# HuggingFace Inference Embeddings
class HFInferenceEmbedding(Embeddings):
    """
    A LangChain-compatible embedding class that calls HuggingFace Inference Providers
    instead of downloading local models.
    """
    def __init__(self, model: str = HF_MODEL, api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get('HF_API_KEY')
        if not self.api_key:
            raise ValueError("HF_API_KEY environment variable not set")

        # HuggingFace official inference client
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=self.api_key
        )

    def embed_query(self, text: str) -> list[float]:
        result = self.client.feature_extraction(text, model=self.model)

        # Normalize output to a 1D list
        if hasattr(result, "tolist"):
            result = result.tolist()

        # Some HF pipelines return list-of-lists
        if isinstance(result[0], list):
            result = result[0]

        return result


    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for t in texts:
            vec = self.client.feature_extraction(t, model=self.model)

            if hasattr(vec, "tolist"):
                vec = vec.tolist()

            if isinstance(vec[0], list):
                vec = vec[0]

            embeddings.append(vec)

        return embeddings



def load_api_embeddings():
    return HFInferenceEmbedding()
