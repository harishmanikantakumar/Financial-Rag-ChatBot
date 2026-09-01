from langchain_community.document_loaders import PyPDFLoader
import os


def load_documents(data_folder):
    """
    Load all PDF files from the given folder.
    """

    documents = []

    # Loop through every file
    for file in os.listdir(data_folder):

        # Only load PDF files
        if file.endswith(".pdf"):

            pdf_path = os.path.join(data_folder, file)

            print(f"Loading: {file}")

            loader = PyPDFLoader(pdf_path)

            docs = loader.load()

            documents.extend(docs)

    return documents