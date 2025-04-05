# RAG-Based Systems

This repository contains implementations of Retrieval-Augmented Generation (RAG) systems using various embedding models and LLMs, including OpenAI, Ollama, and DeepSeek.

## Features
- Load CSV data from AWS S3.
- Create and manage vector-based indices using `llama-index`.
- Query indices using different LLMs and embedding models.
- Support for OpenAI, Ollama, and DeepSeek integrations.

## Prerequisites
- Python 3.8 or higher
- AWS credentials for accessing S3
- OpenAI API key (if using OpenAI models)
- Ollama or DeepSeek setup (if applicable)

## Installation

### Step 1: Set up a virtual environment
1. Create a virtual environment:
   ```bash
   python -m venv env
   ```
2. Activate the virtual environment:
   - On Windows (PowerShell):
     ```bash
     .\env\Scripts\activate.ps1
     ```
   - On Windows (Command Prompt):
     ```bash
     .\env\Scripts\activate.bat
     ```
   - On macOS/Linux:
     ```bash
     source env/bin/activate
     ```

### Step 2: Install dependencies
You can install the required dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Alternatively, you can manually install the dependencies using the following commands:

```bash
pip install boto3
pip install pandas
pip install torch
pip install matplotlib
pip install llama-index
pip install transformers
pip install openai
pip install python-dotenv

pip install llama-index qdrant_client torch transformers
pip install llama-index-llms-ollama
pip install llama-index-embeddings-huggingface
pip install llama-index-llms-deepseek
pip install llama-index-embeddings-ollama
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/hritwij-shrivastava/rag-based-systems.git
   cd rag-based-systems
   ```

2. Set up environment variables:
   - Create a `.env` file in the root directory.
   - Add the following variables:
     ```plaintext
     AWS_ACCESS_KEY=your_aws_access_key
     AWS_SECRET_KEY=your_aws_secret_key
     BUCKET_NAME=your_s3_bucket_name
     OPENAI_API_KEY=your_openai_api_key
     ```

3. Run the desired script:
   - For OpenAI-based RAG:
     ```bash
     python open_ai_rag.py
     ```
   - For Ollama-based RAG:
     ```bash
     python ollama_ai_rag.py
     ```
   - For DeepSeek-based RAG:
     ```bash
     python deepseek_ai_rag.py
     ```

## Notes
- Ensure that the required LLMs (e.g., Ollama or DeepSeek) are running locally or accessible via the specified endpoints.
- Modify the `FILE_NAME` and `STORAGE_PATH` variables in the scripts as needed.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.