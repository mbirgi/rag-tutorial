import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.docstore.document import Document

# Load the API key from env variables
load_dotenv()

RAG_PROMPT_TEMPLATE = """
You are a helpful coding assistant that can answer questions about the provided context. The context is usually a PDF document or an image (screenshot) of a code file. Augment your answers with code snippets from the context if necessary.

If you don't know the answer, say you don't know.

Context: {context}
Question: {question}
"""
PROMPT = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(chunks):
    # Load the embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for the documents
    embeddings = embedding_model.encode([chunk.page_content for chunk in chunks])

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = IndexFlatL2(dimension)
    index.add(embeddings)

    # Create document store
    documents = [Document(page_content=chunk.page_content) for chunk in chunks]
    docstore = InMemoryDocstore(documents)

    # Create index to docstore ID mapping
    index_to_docstore_id = {i: str(i) for i in range(len(chunks))}

    # Create FAISS vector store
    doc_search = FAISS(index, docstore, index_to_docstore_id)
    retriever = doc_search.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )

    # Use Hugging Face pipeline for text generation
    generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')

    def generate_answer(prompt):
        response = generator(prompt, max_length=512, num_return_sequences=1)
        return response[0]['generated_text']

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | generate_answer
        | StrOutputParser()
    )

    return rag_chain

