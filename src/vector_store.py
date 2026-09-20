import os
from langchain_postgres import PGVector
from dotenv import load_dotenv
import hashlib

load_dotenv()
CONNECTION_STRING = f"postgresql+psycopg2://postgres:{os.getenv('DB_PASSWORD')}@localhost:5432/ragdb"

def make_chunk_id(chunk):
    content = chunk.page_content + chunk.metadata.get("source", "")
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def build_vectorstore(chunks, embeddings):
    ids = [make_chunk_id(c) for c in chunks]

    vectorstore = PGVector(
        embeddings=embeddings,
        connection=CONNECTION_STRING,
        collection_name="stage0_docs",
        use_jsonb=True,
    )

    vectorstore.add_documents(documents=chunks, ids=ids)

    return vectorstore