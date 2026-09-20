from langchain_community.vectorstores import PGVector

CONNECTION_STRING = "postgresql+psycopg2://postgres:jiyajiya1234@localhost:5432/ragdb"

def build_vectorstore(chunks, embeddings):
    # Wipe any existing collection with this name first
    #doing this coz it caused the suplication of chunks when i asked a query
    PGVector(
        embedding_function=embeddings,
        connection_string=CONNECTION_STRING,
        collection_name="stage0_docs",
    ).delete_collection()

    return PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection_string=CONNECTION_STRING,
        collection_name="stage0_docs",
    )