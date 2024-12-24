import os
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# HuggingFace API token setup
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HUGGINGFACE_API_TOKEN:
    raise ValueError("Please set HUGGINGFACEHUB_API_TOKEN in your .env file")

class LocalAI:
    def __init__(self, documents_dir: str = "documents", db_dir: str = "db"):
        self.documents_dir = documents_dir
        self.db_dir = db_dir
        self._ensure_directories()
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="result",
            return_messages=True
        )
        
        # Initialize vector store
        self.vector_store = self._initialize_vector_store()
        
        # Initialize conversation chain
        self.qa_chain = self._initialize_qa_chain()

    def _ensure_directories(self):
        """Ensure necessary directories exist and are clean"""
        # Create directories if they don't exist
        os.makedirs(self.documents_dir, exist_ok=True)
        
        # Clean up and recreate db directory
        if os.path.exists(self.db_dir):
            try:
                import shutil
                shutil.rmtree(self.db_dir)
            except Exception as e:
                import psutil
                import signal
                
                # Find processes that might be locking the database
                for proc in psutil.process_iter(['pid', 'name', 'open_files']):
                    try:
                        for file in proc.open_files():
                            if self.db_dir in file.path:
                                os.kill(proc.pid, signal.SIGTERM)
                                break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                # Try cleanup again
                try:
                    shutil.rmtree(self.db_dir)
                except Exception as e:
                    raise ValueError(f"Could not clean up database directory: {str(e)}")
        
        os.makedirs(self.db_dir)

    def _initialize_vector_store(self) -> Chroma:
        """Initialize or load the vector store"""
        try:
            return Chroma(
                persist_directory=self.db_dir,
                embedding_function=self.embeddings
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize vector store: {str(e)}")

    def _initialize_qa_chain(self):
        """Initialize the QA chain with improved configuration"""
        llm = HuggingFaceEndpoint(
            endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
            huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
            task="text-generation",
            temperature=0.7,
            max_new_tokens=512,
            top_p=0.95,
            repetition_penalty=1.15,
            do_sample=True
        )

        prompt_template = """Context: {context}
Question: {question}
Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 4}  # Number of documents to retrieve
            ),
            chain_type_kwargs={
                "prompt": PROMPT,
                "verbose": True
            },
            memory=self.memory,
            return_source_documents=True
        )

    def process_file(self, file_path: str) -> bool:
        """Process and index a single file"""
        try:
            # Select appropriate loader based on file extension
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif ext == '.txt':
                loader = TextLoader(file_path)
            elif ext == '.md':
                loader = UnstructuredMarkdownLoader(file_path)
            elif ext == '.docx':
                loader = Docx2txtLoader(file_path)
            else:
                print(f"Unsupported file type: {ext}")
                return False

            # Load and split the document
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)

            # Add to vector store
            self.vector_store.add_documents(splits)
            self.vector_store.persist()
            return True

        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            return False

    def process_directory(self, directory: str = None) -> None:
        """Process all supported files in a directory"""
        if directory is None:
            directory = self.documents_dir

        supported_extensions = {'.pdf', '.txt', '.md', '.docx'}
        
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in supported_extensions):
                    file_path = os.path.join(root, file)
                    print(f"Processing {file_path}...")
                    self.process_file(file_path)

    def query(self, question: str) -> Dict:
        """Query the knowledge base"""
        try:
            # Call the chain with the question
            chain_output = self.qa_chain({"query": question})  
            
            # Print debug information
            print("Chain output keys:", chain_output.keys())
            print("Chain output:", chain_output)
            
            # Extract answer and sources
            answer = chain_output.get("result", chain_output.get("answer", "No answer found"))
            sources = []
            if "source_documents" in chain_output:
                sources = [doc.page_content for doc in chain_output["source_documents"]]
            
            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            print(f"Debug - Error type: {type(e)}, Error message: {str(e)}")
            return {"error": str(e)}

def main():
    import streamlit as st
    
    st.title("🤖 Local AI Assistant")
    
    # Initialize the AI
    if 'ai' not in st.session_state:
        try:
            st.session_state.ai = LocalAI()
            st.success("AI Assistant initialized successfully!")
        except Exception as e:
            st.error(f"Error initializing AI: {str(e)}")
            return
    # Sidebar for processing documents
    with st.sidebar:
        st.header("Document Processing")
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                try:
                    st.session_state.ai.process_directory()
                    st.success("Processing complete!")
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
    
    # Main chat interface
    st.header("Chat Interface")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.ai.query(prompt)
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                        response_content = f"Error: {result['error']}"
                    else:
                        st.markdown(f"**Answer:** {result['answer']}")
                        if result['sources']:
                            st.markdown("**Sources:**")
                            for i, source in enumerate(result['sources'], 1):
                                st.markdown(f"{i}. {source[:200]}...")
                        response_content = result['answer']
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    response_content = error_msg
                
                st.session_state.messages.append({"role": "assistant", "content": response_content})

if __name__ == "__main__":
    main()
