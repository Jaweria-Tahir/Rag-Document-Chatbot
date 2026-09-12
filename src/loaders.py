# LOADING A DOCUMENT(Document Loader)
from langchain_community.document_loaders import ( PyPDFLoader,TextLoader,UnstructuredMarkdownLoader,
)

#Step 1: setting up uploader of Streamlit.
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

              # selecting loader on baisis of source type
       if original_name.lower().endswith(".pdf"):
              loader = PyPDFLoader(tmp_path)
       elif original_name.lower().endswith(".md"):
              loader = UnstructuredMarkdownLoader(tmp_path)
       else:
              loader = TextLoader(tmp_path)
              # giving doc to loader
       docs = loader.load()
       print(docs)
       # so the metadaata isnot the temp file name
       for doc in docs:
            doc.metadata["source"] = original_name 
            
       st.subheader(f"Loaded: {original_name}")
       st.write(f"Number of Document objects: {len(docs)}")

       for i, doc in enumerate(docs):
            st.write(f"**Chunk/page {i} metadata:**", doc.metadata)
            st.text(doc.page_content)
            st.divider()

       os.remove(tmp_path)