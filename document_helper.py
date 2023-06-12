
import re
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv



def merge_hyphenated_words(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

# In summary, the code replaces any occurrence of a newline character (\n) in the input text, except when it is
# immediately preceded and followed by another newline character, with a single space character. This can be useful for
# removing single line breaks while preserving paragraphs or distinct blocks of text in the content.
def fix_newlines(text: str) -> str:
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)

# the code replaces any occurrence of two or more consecutive newline characters (\n\n, \n\n\n, etc.) in the input text
# with a single newline character (\n).
def remove_multiple_newlines(text: str) -> str:
    return re.sub(r"\n{2,}", "\n", text)


def clean_text(pages, cleaning_functions):
    cleaned_pages = []
    for page_num, metadata, text in pages:
        for cleaning_function in cleaning_functions:
            # all three cleaning_functions are now applied to the text
            text = cleaning_function(text)
        cleaned_pages.append((page_num, metadata, text))
    return cleaned_pages


def text_to_docs(text):
    """Converts list of strings to a list of Documents with metadata."""

    # text = cleaned_text_pdf
    doc_chunks = []

    for page_num, source, page in text:

        if page == '': # in case page is empty
            continue
        # page_num, source, page = text[0][0],text[0][1],text[0][2]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=100,
        )
        chunks = text_splitter.split_text(page)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "page_number": page_num,
                    "chunk": i,
                    "source": source,
                },
            )
            doc_chunks.append(doc)

    return doc_chunks


load_dotenv()
pdf_directory = 'C:/Users/SEPA/custom_chatbot/data/'

def create_vectorstore_embeddings(pdf_directory):
    loader = DirectoryLoader(pdf_directory, glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    raw_pages = []
    for page in documents:
        # print(page.metadata['page']+1)
        # print(page.page_content)
        text = page.page_content
        metadata = page.metadata['source']
        raw_pages.append((page.metadata['page']+1, metadata, text))

    cleaning_functions = [
        merge_hyphenated_words,
        fix_newlines,
        remove_multiple_newlines,
    ]
    cleaned_text_pdf = clean_text(raw_pages, cleaning_functions)
    document_chunks = text_to_docs(cleaned_text_pdf)

    # Step 3 + 4: Generate embeddings and store them in DB
    embeddings = OpenAIEmbeddings(openai_api_key="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")
    vector_store = Chroma.from_documents(
        document_chunks,
        embeddings,
        collection_name="nas-documents",
        persist_directory="data/chroma",
    )

    # Daten wurden jetzt lokal bei uns persistiert
    vector_store.persist()
