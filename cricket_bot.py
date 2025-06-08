"""
Simple Cricket RAG bot using LangChain, Llama2 and ChromaDB.
"""

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def load_documents():
    docs = [
        "Cricket is a bat-and-ball game played between two teams of eleven players.",
        "The ICC Cricket World Cup is the international championship of One Day International cricket.",
        "Sachin Tendulkar holds the record for the most runs in international cricket.",
        "A standard over in cricket consists of six legal deliveries."
    ]
    return docs


def main():
    docs = load_documents()
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_texts(docs, embed_model)
    retriever = db.as_retriever()

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Use the following cricket information to answer the question.\n"
            "Context: {context}\n"
            "Question: {question}\n"
        )
    )

    llm = HuggingFaceHub(repo_id="meta-llama/Llama-2-7b-chat-hf", model_kwargs={"temperature": 0.1})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template}
    )

    while True:
        query = input("Ask a cricket question (or 'quit'): ")
        if query.lower() == "quit":
            break
        answer = qa_chain.run(query)
        print(answer)


if __name__ == "__main__":
    main()
