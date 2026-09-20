from langchain_text_splitters import RecursiveCharacterTextSplitter
# using 1000 as chunk size: 

def split_documents(docs, chunk_size, chunk_overlap):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(docs)

    return chunks




#  Does a chunk cut off mid-sentence in a way that loses meaning?
# Ans: At 1000, the chunk doesnot cut off mid way; but at 300 , it does cut off coz it gives narrow and precise answers


#  If your document has a table, did it get split across two chunks in a way that makes it
# unreadable?
# Ans:This is a known, common limitation of basic PDF text extraction + character-based chunking