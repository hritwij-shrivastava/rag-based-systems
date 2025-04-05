import boto3
import pandas as pd
import os
from io import StringIO
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# AWS Configuration
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# File and storage configuration
FILE_NAME = "Synthea/OHDSI/Gender Vocab/CONCEPT_CLASS.csv"
STORAGE_PATH = "./storage-deepseek-r1"

# LlamaIndex Settings
Settings.embed_model = OllamaEmbedding(
    model_name="deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

Settings.llm = Ollama(
    model="deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    request_timeout=3600.0
)  # Ensure Ollama is running: `ollama run deepseek-r1:1.5b`

def load_csv_from_s3(bucket_name, file_name):
    """
    Load a CSV file from an S3 bucket and return it as a Pandas DataFrame.
    """
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )
    file_obj = s3.get_object(Bucket=bucket_name, Key=file_name)
    df = pd.read_csv(StringIO(file_obj["Body"].read().decode("utf-8")), delimiter="\t")
    df = df.drop("concept_class_id", axis=1)  # Drop unnecessary column
    return df

def create_index_from_csv(df):
    """
    Create a LlamaIndex from a Pandas DataFrame.
    """
    # Convert DataFrame rows into structured text documents
    docs = [
        Document(
            text=f"concept_class_name: {row['concept_class_name']}, concept_class_concept_id: {row['concept_class_concept_id']}"
        )
        for _, row in df.iterrows()
    ]

    # Parse documents into nodes
    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(docs)

    # Build and return the index
    return VectorStoreIndex(nodes, embed_model=Settings.embed_model)

def main():
    """
    Main execution function to load data, create an index, and query it.
    """
    # Load CSV data from S3
    print("Loading CSV data from S3...")
    csv_data = load_csv_from_s3(BUCKET_NAME, FILE_NAME)

    # Create an index from the CSV data
    print("Creating index from CSV data...")
    index = create_index_from_csv(csv_data)

    # Query the index
    print("Querying the index...")
    query_engine = index.as_query_engine(llm=Settings.llm)
    query = "Show me concept_class_concept_id for Quality Metric."
    retrieved_data = query_engine.query(query)

    # Display the retrieved data
    print("\nRetrieved Data:")
    print(retrieved_data)

if __name__ == "__main__":
    main()