import os
import argparse
from tqdm import tqdm
import torch
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import UnstructuredFileLoader

from langchain_community.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

parser = argparse.ArgumentParser()
parser.add_argument("--docs_dir", "-d", help="Directory for RAG reference documents.", type=str, default="./docs/demo")
parser.add_argument("--persist_dir", "-p", help="Persistent directory for the vectorDB.", type=str, default="./vectorDB/demo")
parser.add_argument("--embedder", help="Directory for the embedder model.", type=str, default="../autodl-tmp/model/bge-m3")
parser.add_argument("--file_types", type=str, default="md,txt")
parser.add_argument("--chunk_size", type=int, default=100)
args = parser.parse_args()


def get_files(dir_path, types=[".md"]):
    """
    Args:
        dir_path (str): Directory for RAG docs
        types (list, optional): File types of RAG docs. Only support markdown and csv. Defaults to [".md"].

    Returns:
        _type_: list of file paths.
    """
    file_list = []
    for filepath, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            for file_type in types:
                if filename.endswith(file_type):
                    file_list.append(os.path.join(filepath, filename))
                    continue

    return file_list


def get_text(dir_path, types=[".md"]):
    """
    Args:
        dir_path (str): Directory for RAG docs
        types (list, optional): File types of RAG docs. Only support markdown and csv. Defaults to [".md"].

    Returns:
        _type_: list of docs.
    """
    file_list = get_files(dir_path, types)
    docs = []
    for one_file in tqdm(file_list):
        file_type = one_file.split('.')[-1]
        if file_type == 'md':
            loader = UnstructuredMarkdownLoader(one_file)
        elif file_type == 'txt':
            loader = UnstructuredFileLoader(one_file)
        else:
            continue
        docs.append(loader.load())
    return docs


    
def main():
    
    embedder = HuggingFaceEmbeddings(model_name=args.embedder)
    embedder.client.requires_grad_(False)
    
    tar_dir = args.docs_dir
    file_types = args.file_types.split(",")
    docs_list = get_text(tar_dir, file_types)

    persist_directory = args.persist_dir
    
    try:
        vectordb = Chroma(
        persist_directory=persist_directory, 
        # embedding_function=embeddings
        )
        vectordb.delete_collection()
        print("Successfully initialize the ChromaDB.")
        print(f"Vector database initialized at {persist_directory}.")
    except:
        raise Exception("Initilization Error!")
    
    vectordb = Chroma(
        embedding_function=embedder,
        persist_directory=persist_directory 
    )
    
    split_docs = []
    for doc_sublist in tqdm(docs_list):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n"], chunk_size=args.chunk_size, chunk_overlap=25,keep_separator=False)
        # split_docs.extend(text_splitter.split_documents(doc_sublist))
        vectordb.add_documents(text_splitter.split_documents(doc_sublist))
        vectordb.persist()

    
if __name__ == "__main__":
    main()