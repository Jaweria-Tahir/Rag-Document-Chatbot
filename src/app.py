import streamlit as st
from loaders import load_document
from text_splitter import split_documents
from text_embeddings import get_embeddings
from vector_store import build_vectorstore

# doc uploader by streamlit
uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True)

if uploaded_files:
    all_chunks = []

    for u in uploaded_files:
        # loader
        docs = load_document(u)
        # chunking
        chunks = split_documents(docs, chunk_size=1000, chunk_overlap=200)
        all_chunks.extend(chunks)

    st.success(f"Loaded and chunked {len(all_chunks)} chunks total")

    # embeddings & making vector store / external knowledge base
    if "vectorstore" not in st.session_state:
        with st.spinner("Embedding chunks and storing in Postgres..."):
            embeddings = get_embeddings()
            st.session_state.vectorstore = build_vectorstore(all_chunks, embeddings)
        st.success("Vectorstore built!")
#same document/chunk from being inserted twice.
    vectorstore = st.session_state.vectorstore

    # raw similarity search; no LLM
    st.subheader("Test Retrieval without LLM")
    query = st.text_input("Ask a test question about your document")

    if query:
        results = vectorstore.similarity_search(query, k=3)

        st.write(f"Top {len(results)} matching chunks:")
        for i, r in enumerate(results):
            st.write(f"**Result {i+1}** — source: {r.metadata.get('source')}, page: {r.metadata.get('page')}")
            st.text(r.page_content)
            st.divider()