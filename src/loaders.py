# LOADING A DOCUMENT (Document Loader)
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, PyMuPDFLoader,
)
import os
import tempfile


def load_document(uploaded_file):
    original_name = uploaded_file.name  # Save for citation in the response.

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(original_name)[1]
    ) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # selecting loader on basis of source type
    if original_name.lower().endswith(".pdf"):
        loader = PyMuPDFLoader(tmp_path)
    elif original_name.lower().endswith(".md"):
        loader = TextLoader(tmp_path, encoding="utf-8")
    else:
        loader = TextLoader(tmp_path)

    docs = loader.load()

    # fix metadata so it's not the temp file name
    for doc in docs:
        doc.metadata["source"] = original_name
        # NEW: strip NUL characters and any other problematic control chars
        doc.page_content = doc.page_content.replace("\x00", "")

    os.remove(tmp_path)
    return docs