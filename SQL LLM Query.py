from llama_index.core import (
    Settings,
    VectorStoreIndex,
)
import sqlite3

from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from dotenv import load_dotenv
import os
import toml


from llama_index.llms.huggingface import HuggingFaceLLM

from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Document,
    Settings,
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI    

from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore


load_dotenv()
#model_config['protected_namespaces'] = ()
llm_deployment = os.getenv("AZURE_LLM_DEPLOYMENT")
embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_version = os.getenv("API_VERSION")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_port = os.getenv("QDRANT_PORT")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

config = toml.load("config.toml")
SIMILARITY_TOP_K = config["retrieval"]["similarity_top_k"]
QDRANT_COLLECTION_NAME = config["vectordb"]["collection_name"]


# bge-base embedding model
embed_model= AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name=embedding_deployment,
    api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    api_version=azure_api_version,
)
Settings.embed_model = embed_model

llm = AzureOpenAI(
    model="gpt-4o",
    deployment_name=llm_deployment,
    api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    api_version=azure_api_version,
)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")


Settings.llm = llm
Settings.embed_model = embed_model

documents = SimpleDirectoryReader("dummy").load_data()
index = VectorStoreIndex.from_documents(documents)



def Process_LLM(query):
    db_path = f'split_Files/ERS.sql'  # Replace with your desired database name
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query_engine = index.as_query_engine(similarity_top_k=SIMILARITY_TOP_K)
    #query = "Please find name with id = 51346771"
    table_name = f'ERS_HR'  # Replace with your desired table name
    # Get the column names
    # Execute a query
    cursor.execute(f"SELECT * FROM {table_name}")
    column_names = [description[0] for description in cursor.description]


    query_main = f"from column names {column_names} and from table {table_name} Please generate only sql query without any additional text for " + query
    response = query_engine.query(query_main)

    # Remove the unwanted tag (e.g., '''sql)
    cleaned_response = str(response).replace("sql", "").strip()  # Remove the tag and strip extra spaces/newlines
    cleaned_response = cleaned_response.replace("```", "").strip() 

    db_path = f'split_Files/ERS.sql'  # Replace with your desired database name

    # Define a table name for your data
    table_name = f'ERS_HR'  # Replace with your desired table name

    cursor = conn.cursor()

    # Get column names dynamically

    output = cursor.execute(cleaned_response)
    s = output.fetchall()
    
    response = query_engine.query(f"Write a detailed executive writeup with very good formatting , about this result of an SQL query mentioning all the data fields and their values  from SQL DB- {s}.  Also mention the table formatted data. Use the query {query} for generating this data. But, Dont mention {query} in the response")
    conn.commit()
    conn.close()
    return(response)


query = "Along with metadata, Calculate the percentage of employees working in every country against total employee count. "
response = Process_LLM(query)
print(response)

