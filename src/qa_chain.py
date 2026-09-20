#User question → PGVector finds relevant chunks → Llama 3 uses those chunks to generate the answer → source chunks are also returned.

from langchain_classic.chains import RetrievalQA
from langchain_ollama import OllamaLLM

def get_llm():
    return OllamaLLM(model="qwen2.5:1.5b")

def build_qa_chain(vectorstore):
    llm = get_llm()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    return qa_chain
