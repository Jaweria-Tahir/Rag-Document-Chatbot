from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(docs, chunk_size=1000, chunk_overlap=200):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(docs)

    return chunks