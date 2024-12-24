# Local Document AI Assistant

A powerful local AI assistant that can process and understand your documents, providing intelligent responses based on your data. All processing happens locally, ensuring your data remains private.

## Features
- Process multiple document types (PDF, TXT, MD, DOCX)
- Local document storage and processing
- Efficient vector-based document retrieval
- Conversational interface for document queries
- Privacy-focused (all data stays on your machine)

## Setup
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up HuggingFace API token:
- Get your token from https://huggingface.co/settings/tokens
- Create a `.env` file in the project root
- Add your token: `HUGGINGFACE_API_TOKEN=your_token_here`

4. Create document directory:
- Place your documents in the `documents` folder

## Usage
Run the assistant:
```bash
python main.py
```

Commands:
- `process`: Process all documents in the documents directory
- `query`: Enter query mode to ask questions about your documents
- `exit`: Exit the program

## Supported File Types
- PDF (.pdf)
- Text files (.txt)
- Markdown (.md)
- Word documents (.docx)

## How it Works
1. Documents are processed and split into chunks
2. Text chunks are converted to embeddings using sentence-transformers
3. Embeddings are stored in a local Chroma vector database
4. Questions are processed using the same embedding model
5. Relevant document chunks are retrieved and used to generate answers

## Privacy
All processing happens locally on your machine. No data is sent to external services except for the model inference which uses HuggingFace's API.
