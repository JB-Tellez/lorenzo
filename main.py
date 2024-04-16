from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


import json
from pathlib import Path
from pprint import pprint

embeddings = OllamaEmbeddings()

loader = WebBaseLoader("https://reedsy.com/discovery/blog/best-fantasy-books")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)



llm = Ollama(model="llama2")

prompt = ChatPromptTemplate.from_template("""Recommend a book on list:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

response = retrieval_chain.invoke({"input": "Any books about leadership and facing adversity?"})
print(response["answer"])
