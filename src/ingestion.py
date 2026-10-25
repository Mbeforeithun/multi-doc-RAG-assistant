from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
    ".csv": CSVLoader,
}


# def load_documents(datadirectory = "data"):
#     all_documents = []

#     for path in Path(datadirectory).iterdir():

#         if not path.is_file():
#             continue

#         if path.suffix.lower() not in LOADERS:
#             continue

#         all_documents.extend(load_document(path))

#     return all_documents

def load_documents(data_dir="data", only_files=None):
    all_documents = []

    for path in Path(data_dir).iterdir():
        if not path.is_file() or path.suffix.lower() not in LOADERS:
            continue
        if only_files is not None and path.name not in only_files:
            continue
        all_documents.extend(load_document(path))

    return all_documents    


def load_document(path):
    path = Path(path)

    extension = path.suffix.lower()

    if extension not in LOADERS:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    loader_class = LOADERS[extension]

    documents = loader_class(str(path)).load()

    for document in documents:
        document.metadata["source"] = path.name

    return documents

# ---------- Chunking ----------

def split_documents(documents, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


# ---------- Embeddings + vector store ----------
INDEX_DIR = "faiss_index"
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )

def get_vectorstore():
    """Load the saved index, or return None if nothing has been indexed yet."""
    if Path(INDEX_DIR).exists():
        return FAISS.load_local(
            INDEX_DIR,
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return None

import json

INDEXED_FILES_LOG = "indexed_files.json"


def get_indexed_files():
    if Path(INDEXED_FILES_LOG).exists():
        return set(json.loads(Path(INDEXED_FILES_LOG).read_text()))
    return set()


def mark_files_indexed(filenames):
    indexed = get_indexed_files()
    indexed.update(filenames)
    Path(INDEXED_FILES_LOG).write_text(json.dumps(list(indexed)))




def add_to_vectorstore(chunks):
    vectorstore = get_vectorstore()
    if vectorstore is None:
        vectorstore = FAISS.from_documents(chunks, get_embeddings())
    else:
        vectorstore.add_documents(chunks)
    vectorstore.save_local(INDEX_DIR)
    return vectorstore

# ---------- Quick test ----------

if __name__ == "__main__":
    TEST_QUESTION = "What do adaptive approaches to digital ecosystems begin with?"  # replace with a real question from your data

    print("STEP 1: Loading documents...", flush=True)
    docs = load_documents()
    print(f"Loaded {len(docs)} documents", flush=True)

    print("STEP 2: Splitting documents...", flush=True)
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks", flush=True)

    print("STEP 3: Embedding and storing in FAISS...", flush=True)
    vectorstore = add_to_vectorstore(chunks)
    print("Stored and saved to disk", flush=True)

    print(f"STEP 4: Searching for: {TEST_QUESTION}", flush=True)
    for r in vectorstore.similarity_search(TEST_QUESTION, k=3):
        print(r.metadata, "->", r.page_content[:150], "\n", flush=True)