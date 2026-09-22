#User question → PGVector finds relevant chunks → Llama 3 uses those chunks to generate the answer → source chunks are also returned.

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaLLM
from memory import get_memory

def get_llm():
    return OllamaLLM(model="qwen2.5:1.5b")

def build_qa_chain(vectorstore):
    llm = get_llm()
    memory = get_memory()

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )
    return qa_chain

#using ConversationalRetrievalChain + ConversationBufferMemory , over retreivalQA it remembers past turns, and rewrites your follow-up into a standalone question using that history before searching the vectorstore