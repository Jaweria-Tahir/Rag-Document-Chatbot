import streamlit as st
from loaders import load_document
from text_splitter import split_documents
from text_embeddings import get_embeddings
from vector_store import build_vectorstore
from qa_chain import build_qa_chain

# doc uploader by streamlit
uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True)

if uploaded_files:
    all_chunks = []

    for u in uploaded_files:
        # step 1: loader
        docs = load_document(u)
        # step 2: chunking
        chunks = split_documents(docs, chunk_size=1000, chunk_overlap=200)
        all_chunks.extend(chunks)

    st.success(f"Loaded and chunked {len(all_chunks)} chunks total")

    # step 3: embeddings & making vector store / external knowledge base
    #New files → embed + store → remember vectorstore → reuse it without rebuilding every time.
    if "last_uploaded_names" not in st.session_state or st.session_state.last_uploaded_names != [u.name for u in uploaded_files]:
        with st.spinner("Embedding chunks and storing in Postgres..."):
            embeddings = get_embeddings()
            vectorstore = build_vectorstore(all_chunks, embeddings)
        st.session_state.vectorstore = vectorstore #session_state lets Streamlit remember something between interactions/reruns.
        st.session_state.last_uploaded_names = [u.name for u in uploaded_files]
        st.success("Vectorstore updated!")

    vectorstore = st.session_state.vectorstore


    # raw similarity search; no LLM
    #vectore serach using similarity doesnot give exact chunks , it only helps in narrowing down
    #but an LLM understands and gives accurate info 
    
    # st.subheader("Test Retrieval without LLM")
    # query = st.text_input("Ask a test question about your document")

# step 4: retriever + augmentation + genartion through LLM
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = build_qa_chain(vectorstore)
    qa_chain = st.session_state.qa_chain

    st.subheader("Ask a question (with LLM answer)")
    query = st.text_input("Your question")

    if query:
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"query": query})

        st.write("### Answer")
        st.write(result["result"])

        st.write("### Source chunks used")
        for i, doc in enumerate(result["source_documents"]):
            st.write(f"**Source {i+1}** — {doc.metadata.get('source')}, page {doc.metadata.get('page')}")
            st.text(doc.page_content)
            st.divider()