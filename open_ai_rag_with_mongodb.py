import boto3
import pandas as pd
import os
from io import StringIO
import pymongo
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import Document, Settings
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load environment variables
OpenAI.api_key = os.getenv("OPENAI_API_KEY")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")
MONGO_URI = os.getenv("MONGO_URI")
ENV = os.getenv("ENV", "prod")  # Default to "prod" if ENV is not set

# MongoDB configuration
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME")
FILE_NAME = os.getenv("FILE_NAME")

# Embedding and LLM configuration
embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=256)
llm = OpenAI()

Settings.llm = llm
Settings.embed_model = embed_model


def get_mongo_client(mongo_uri):
    """Establish connection to the MongoDB."""
    try:
        client = pymongo.MongoClient(mongo_uri)
        print("Connection to MongoDB successful")
        return client
    except pymongo.errors.ConnectionFailure as e:
        print(f"Connection failed: {e}")
        return None


def load_csv_from_s3(bucket_name, file_name):
    """Load CSV file from S3."""
    s3 = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
    file_obj = s3.get_object(Bucket=bucket_name, Key=file_name)
    df = pd.read_csv(StringIO(file_obj["Body"].read().decode("utf-8")), delimiter="\t")
    df = df.drop("concept_class_id", axis=1)
    return df

def get_vector_store(mongodb_client):
    """Get the MongoDB vector store."""
    vector_store = MongoDBAtlasVectorSearch(
        mongodb_client=mongodb_client,
        db_name=MONGO_DB_NAME,
        collection_name=MONGO_COLLECTION_NAME,
        vector_index_name="vector_index"
    )
    return vector_store

def create_index_from_csv(df, vector_store):
    """Create a vector index from the CSV data."""
    docs = []
    for _, row in df.iterrows():
        text = f"concept_class_name: {row['concept_class_name']}, concept_class_concept_id: {row['concept_class_concept_id']}"
        docs.append(Document(text=text))
        
    # Convert to LlamaIndex format
    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(docs)

    # Configure MongoDBAtlasVectorSearch as the vector store
    vector_store = MongoDBAtlasVectorSearch(
        mongodb_client=mongodb_client,
        db_name=MONGO_DB_NAME,
        collection_name=MONGO_COLLECTION_NAME,
        vector_index_name="vector_index"  # You have to create this index in MongoDB
    )

    # Add nodes to the vector store
    for node in nodes:
        node_embedding = embed_model.get_text_embedding(node.get_content(metadata_mode="all"))
        node.embedding = node_embedding
    
    vector_store.add(nodes)
    index = VectorStoreIndex.from_vector_store(vector_store)
    return index


def load_existing_index(vector_store):
    """Load an existing vector index from MongoDB."""
    index = VectorStoreIndex.from_vector_store(vector_store)
    return index


# Main Execution
if __name__ == "__main__":
    mongodb_client = get_mongo_client(MONGO_URI)
    vector_store = get_vector_store(mongodb_client)

    if ENV == "train":
        print("Environment: TRAIN")
        # Load CSVs from S3
        csv_data = load_csv_from_s3(BUCKET_NAME, FILE_NAME)
        # Create and store the vector index
        index = create_index_from_csv(csv_data, vector_store)
        print("Vector index created successfully.")
    elif ENV == "prod":
        print("Environment: PROD")
        # Load the existing vector index
        index = load_existing_index(vector_store)
        print("Vector index loaded successfully.")
        
        # Query the Index
        query_engine = index.as_query_engine()
        query = "what is the concept_class_concept_id for Quality Metric."
        retrieved_data = query_engine.query(query)
        
        print("\nRetrieved Data:")
        print(retrieved_data)
    else:
        print(f"Unknown environment: {ENV}")