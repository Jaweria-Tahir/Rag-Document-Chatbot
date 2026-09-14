# LOADING A DOCUMENT(Document Loader)
from langchain_community.document_loaders import (PyPDFLoader, TextLoader, UnstructuredMarkdownLoader,
)
from text_splitter import split_documents
# Step 1: setting up uploader of Streamlit.
import os
import tempfile
import streamlit as st

uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True)

# Step 2: creating a temporary file for each uploaded file & saving name of orginal file
for u in uploaded_files:
       original_name = u.name  # Save for citation in the response.

       with tempfile.NamedTemporaryFile(
              delete=False, suffix=os.path.splitext(original_name)[1]
       ) as tmp:
              tmp.write(u.read())
              tmp_path = tmp.name

              # selecting loader on baisis of source type & giving doc to loader
       if original_name.lower().endswith(".pdf"):
              loader = PyPDFLoader(tmp_path)
       elif original_name.lower().endswith(".md"):
              loader = TextLoader(tmp_path, encoding="utf-8")
       else:
              loader = TextLoader(tmp_path)
#step 3: loader loads the text of the file
       docs = loader.load()
       # print(docs)
       
       #SPLITTER
       chunks = split_documents(
              docs,
              chunk_size=1000,
              chunk_overlap=200
              )
       st.subheader("Sample Chunks")
       for i, chunk in enumerate(chunks[:3]):
              st.write(f"### Chunk {i + 1}")
              st.text(repr(docs[0].page_content))
              st.text(chunk.page_content)
              st.divider()
              
              
       # so the metadaata isnot the temp file name
       for doc in docs:
            doc.metadata["source"] = original_name 
            
       st.subheader(f"Loaded: {original_name}")
       st.write(f"Number of Document objects: {len(docs)}")

       for i, doc in enumerate(docs):
            st.write(f"**Chunk/page {i} metadata:**", doc.metadata)
            st.text(doc.page_content)
            st.divider()
# step 4: delete temp file
       os.remove(tmp_path)