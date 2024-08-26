import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from utils import load_config
from langchain_community.embeddings import GooglePalmEmbeddings
from termcolor import colored

# Load the configuration
cfg = load_config()
os.environ['GOOGLE_API_KEY'] =  cfg.PALLM_API

def load_dir_data():
    """Loads PDF documents from the specified directory."""
    loader = DirectoryLoader(
        cfg.DATA_PATH,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )
    document_data = []
    try:
        document_data = loader.load()
    except Exception as e:
        print(f"Error loading files: {e}")

    return document_data

def split_doc_to_chunks(document_data):
    """Splits the loaded documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE, chunk_overlap=cfg.CHUNK_OVERLAP
    )
    all_splits = text_splitter.split_documents(document_data)
    return all_splits

def save_data_to_faiss_vector_db(all_splits):
    embeddings = GooglePalmEmbeddings()
    vectorstore = FAISS.from_documents(all_splits, embeddings)
    print(all_splits)
    print("############################")
    print(vectorstore)
    vectorstore.save_local(cfg.DB_FAISS_PATH)

    return "Database build completed ..."


def load_data():
    """Loads data, splits it into chunks, and saves it to the vector database."""
    document_data = load_dir_data()
    all_splits = split_doc_to_chunks(document_data)
    return save_data_to_faiss_vector_db(all_splits)

def init_vector_db():
    if not os.path.exists(cfg.DB_FAISS_PATH):
        os.makedirs(cfg.DB_FAISS_PATH)
        print(colored(f"{cfg.DB_FAISS_PATH} directory created.", "blue"))

    files = [file for file in os.listdir(cfg.DB_FAISS_PATH) if file != "readme.md"]

    if not files:
        print(
            colored(
                f"{cfg.DB_FAISS_PATH} is empty. Build vector DB on first run", "red"
            )
        )
        resp = load_data()
        print(resp)
    else:
        print(
            colored(
                f"{cfg.DB_FAISS_PATH} is not empty. No need to build vector DB on first run",
                "green",
            )
        )

# Entry point for the script
if __name__ == "__main__":
    init_vector_db()
