import streamlit as st
from loaders import load_document
from text_splitter import split_documents
from text_embeddings import get_embeddings
from vector_store import build_vectorstore
from qa_chain import build_qa_chain

st.title("Document Q&A Chat")

# doc uploader by streamlit
uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True)

if uploaded_files:
    # track which files are already ingested, so re-running doesn't re-embed everything
    if "ingested_files" not in st.session_state:
        st.session_state.ingested_files = set()

    new_files = [u for u in uploaded_files if u.name not in st.session_state.ingested_files]

    if new_files:
        all_chunks = []
        for u in new_files:
            # step 1: loader
            docs = load_document(u)
            # step 2: chunking
            chunks = split_documents(docs, chunk_size=1000, chunk_overlap=200)
            all_chunks.extend(chunks)
            st.session_state.ingested_files.add(u.name)

        # step 3: embeddings & vectorstore (ADDS to existing store, doesn't replace)
        with st.spinner(f"Embedding {len(all_chunks)} chunks and storing in Postgres..."):
            embeddings = get_embeddings()
            vectorstore = build_vectorstore(all_chunks, embeddings)

        st.session_state.vectorstore = vectorstore
        st.success(f"Added {len(new_files)} new file(s): {', '.join(u.name for u in new_files)}")
        
        
        #DEBUGGING RETREIEVAL
# if "vectorstore" in st.session_state:
#     st.subheader("DEBUG: Raw retrieval test")
#     debug_query = st.text_input("Debug question")
#     if debug_query:
#         debug_results = st.session_state.vectorstore.similarity_search(debug_query, k=3)
#         for i, r in enumerate(debug_results):
#             st.write(f"**Chunk {i+1}** — {r.metadata.get('source')}, page {r.metadata.get('page')}")
#             st.text(r.page_content)
#             st.divider()

# step 4 + 5: qa_chain (retrieval + LLM + memory), built once vectorstore exists
if "vectorstore" in st.session_state:
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = build_qa_chain(st.session_state.vectorstore)
    qa_chain = st.session_state.qa_chain

    # chat message history (for display)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # replay past messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Source chunks used"):#sources in expander
                    for i, src in enumerate(msg["sources"]):
                        st.write(f"**Source {i+1}** — {src['source']}, page {src['page']}")
                        st.text(src["content"])
                        st.divider()

    # step 6: chat input instead of plain text_input
    query = st.chat_input("Ask a question about your documents")

    if query:
        with st.chat_message("user"):
            st.write(query)
        st.session_state.messages.append(
            {"role": "user", "content": query}
            )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = qa_chain.invoke({"question": query})
                answer = result["answer"]
                sources = [
                    {
                        "source": doc.metadata.get("source"),
                        "page": doc.metadata.get("page"),
                        "content": doc.page_content
                    }
                    for doc in result["source_documents"]
                ]
            st.write(answer)
            with st.expander("Source chunks used"):
                for i, src in enumerate(sources):
                    st.write(f"**Source {i+1}** — {src['source']}, page {src['page']}")
                    st.text(src["content"])
                    st.divider()

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
else:
    st.info("Upload at least one document to start chatting.")