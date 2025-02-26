from langchain_graph_retriever import GraphRetriever
import os
from langchain_openai import OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from dotenv import load_dotenv 
from graph_retriever.strategies import Eager
load_dotenv()

ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_NAMESPACE = os.environ.get("ASTRA_DB_NAMESPACE")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = AstraDBVectorStore(
    collection_name="test",
    embedding=embeddings,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,    
    autodetect_collection=True,
)

retriever = GraphRetriever(
    store = vector_store,    
    edges = [("metadata.entities", "metadata.entities")],
    strategy = Eager(k=10,start_k=3,max_depth=2),
)

docs = retriever.invoke("your question here")
print(docs)

