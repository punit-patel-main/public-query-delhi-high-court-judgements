import os
import pandas as pd
from typing import List, Dict, Any
import logging
from tqdm import tqdm
from dotenv import load_dotenv
from typing import Optional

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LangChainCourtJudgmentProcessor:
    def __init__(self, pinecone_api_key: str, pinecone_index_name: str = "court-judgments"):
        self.pinecone_api_key = pinecone_api_key
        self.index_name = pinecone_index_name
        
        # Initialize embeddings model
        # Models = ['all-MiniLM-L6-v2', 'BAAI/bge-base-en', 'intfloat/e5-base-v2']
        model = 'all-MiniLM-L6-v2'
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize text splitter with sliding window approach
        # Note: We'll use approximate character counts (avg 5 chars per word)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # ~300 words * 5 chars/word
            chunk_overlap=250,  # ~50 words * 5 chars/word
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=pinecone_api_key)
        # self.vectorstore = None
        
        # Load metadata CSV
        self.metadata_df = None
        
    def setup_pinecone_index(self):
        """Create or connect to Pinecone index"""
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes()
            index_names = [index.name for index in existing_indexes]
            
            if self.index_name not in index_names:
                # Create index with appropriate dimensions for all-MiniLM-L6-v2 (384 dimensions)
                self.pc.create_index(
                    name=self.index_name,
                    dimension=384,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'  # Adjust region as needed
                    )
                )
                logger.info(f"Created new Pinecone index: {self.index_name}")
            
            # Initialize LangChain Pinecone vectorstore
            self.vectorstore = PineconeVectorStore.from_existing_index(
                index_name=self.index_name,
                embedding=self.embeddings
            )
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error setting up Pinecone index: {e}")
            raise
    
    def load_metadata_csv(self, csv_path: str):
        """Load the metadata CSV file"""
        try:
            self.metadata_df = pd.read_csv(csv_path)
            logger.info(f"Loaded metadata CSV with {len(self.metadata_df)} rows")
            
            # Verify required columns exist
            required_columns = ['link_id', 'caseNo', 'party']
            missing_columns = [col for col in required_columns if col not in self.metadata_df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in CSV: {missing_columns}")
                
        except Exception as e:
            logger.error(f"Error loading metadata CSV: {e}")
            raise
    
    def get_metadata_for_pdf(self, pdf_filename: str) -> Dict[str, List[str]]:
        """
        Get case numbers and party names for a PDF from metadata CSV
        
        Returns:
            Dictionary with case_numbers and party_names lists
        """
        if self.metadata_df is None:
            return {'case_numbers': [], 'party_names': []}
        
        # Remove file extension from filename for matching
        link_id = os.path.splitext(pdf_filename)[0]
        print(link_id)
        
        # Filter rows matching the link_id
        matching_rows = self.metadata_df[self.metadata_df['link_id'].astype(str) == str(link_id)]
        
        if matching_rows.empty:
            logger.warning(f"No metadata found for PDF: {pdf_filename}")
            return {'case_numbers': [], 'party_names': []}
        
        # Extract unique case numbers and party names
        case_numbers = matching_rows['caseNo'].dropna().unique().tolist()
        party_names = matching_rows['party'].dropna().unique().tolist()
        
        # Clean up the arrays (remove empty strings)
        case_numbers = [str(case).strip() for case in case_numbers if str(case).strip()]
        party_names = [str(party).strip() for party in party_names if str(party).strip()]
        
        return {'case_numbers': case_numbers, 'party_names': party_names}
    
    # def load_and_process_pdf(self, pdf_path: str) -> List[Document]:
    #     """
    #     Load PDF and create processed documents with metadata
        
    #     Returns:
    #         List of LangChain Document objects with enhanced metadata
    #     """
    #     pdf_filename = os.path.basename(pdf_path)
    #     logger.info(f"Processing PDF: {pdf_filename}")
        
    #     try:
    #         # Load PDF using LangChain's PyPDFLoader
    #         loader = PyPDFLoader(pdf_path)
    #         pages = loader.load()
            
    #         if not pages:
    #             logger.warning(f"No pages loaded from {pdf_filename}")
    #             return []
            
    #         # Get metadata for this PDF
    #         pdf_metadata = self.get_metadata_for_pdf(pdf_filename)
            
    #         # Combine all pages into one document for chunking
    #         full_text = "\n\n".join([page.page_content for page in pages])
            
    #         # Create a single document with all text
    #         combined_doc = Document(
    #             page_content=full_text,
    #             metadata={
    #                 'source': pdf_path,
    #                 'pdf_filename': pdf_filename,
    #                 'total_pages': len(pages),
    #                 **pdf_metadata
    #             }
    #         )
            
    #         # Split into chunks
    #         chunks = self.text_splitter.split_documents([combined_doc])
            
    #         # Enhance each chunk with additional metadata
    #         enhanced_chunks = []
    #         for i, chunk in enumerate(chunks):
    #             # Calculate approximate page range for chunk
    #             chunk_position = i / len(chunks)
    #             start_page = max(1, int(chunk_position * len(pages)))
    #             end_page = min(len(pages), int((chunk_position + 1/len(chunks)) * len(pages)) + 1)
                
    #             # Create judgment_id
    #             judgment_id = f"{os.path.splitext(pdf_filename)[0]}_chunk_{i+1}"
                
    #             # Update chunk metadata
    #             chunk.metadata.update({
    #                 'judgment_id': judgment_id,
    #                 'start_page': start_page,
    #                 'end_page': end_page,
    #                 'chunk_index': i + 1,
    #                 'total_chunks': len(chunks)
    #             })
                
    #             enhanced_chunks.append(chunk)
            
    #         logger.info(f"Created {len(enhanced_chunks)} chunks from {pdf_filename}")
    #         return enhanced_chunks
            
    #     except Exception as e:
    #         logger.error(f"Error processing {pdf_filename}: {e}")
    #         return []

    def load_and_process_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load PDF and create processed documents with metadata
        
        Returns:
            List of LangChain Document objects with enhanced metadata
        """
        pdf_filename = os.path.basename(pdf_path)
        logger.info(f"Processing PDF: {pdf_filename}")
        
        try:
            # Load PDF using LangChain's PyPDFLoader
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            if not pages:
                logger.warning(f"No pages loaded from {pdf_filename}")
                return []
            
            # Get metadata for this PDF
            pdf_metadata = self.get_metadata_for_pdf(pdf_filename)
            
            # Create documents for each page with page tracking
            page_documents = []
            for page_num, page in enumerate(pages, 1):
                page_doc = Document(
                    page_content=page.page_content,
                    metadata={
                        'source': pdf_path,
                        'pdf_filename': pdf_filename,
                        'page_number': page_num,
                        'total_pages': len(pages),
                        **pdf_metadata
                    }
                )
                page_documents.append(page_doc)
            
            # Split each page into chunks while preserving page information
            all_chunks = []
            chunk_counter = 1
            
            for page_doc in page_documents:
                # Split this page into chunks
                page_chunks = self.text_splitter.split_documents([page_doc])
                
                # Process each chunk from this page
                for chunk in page_chunks:
                    # Create judgment_id
                    judgment_id = f"{os.path.splitext(pdf_filename)[0]}_chunk_{chunk_counter}"
                    
                    # Update chunk metadata with exact page number
                    chunk.metadata.update({
                        'judgment_id': judgment_id,
                        'start_page': page_doc.metadata['page_number'],
                        'end_page': page_doc.metadata['page_number'],
                        'chunk_index': chunk_counter,
                        'page_number': page_doc.metadata['page_number']  # Keep original page number
                    })
                    
                    all_chunks.append(chunk)
                    chunk_counter += 1
            
            # Update total_chunks for all chunks
            for chunk in all_chunks:
                chunk.metadata['total_chunks'] = len(all_chunks)
            
            logger.info(f"Created {len(all_chunks)} chunks from {pdf_filename}")
            return all_chunks
            
        except Exception as e:
            logger.error(f"Error processing {pdf_filename}: {e}")
            return []
    
    def process_single_pdf(self, pdf_path: str) -> List[Document]:
        """Process a single PDF and return Document objects"""
        return self.load_and_process_pdf(pdf_path)
    
    def add_documents_to_vectorstore(self, documents: List[Document], batch_size: int = 100):
        """Add documents to Pinecone vectorstore in batches"""
        try:
            total_docs = len(documents)
            logger.info(f"Adding {total_docs} documents to vectorstore")
            
            for i in range(0, total_docs, batch_size):
                batch = documents[i:i + batch_size]
                
                # Extract texts and metadatas for batch processing
                texts = [doc.page_content for doc in batch]
                metadatas = [doc.metadata for doc in batch]
                ids = [doc.metadata['judgment_id'] for doc in batch]
                
                # Add to vectorstore
                self.vectorstore.add_texts(
                    texts=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                
                logger.info(f"Added batch {i//batch_size + 1}/{(total_docs-1)//batch_size + 1}")
                
        except Exception as e:
            logger.error(f"Error adding documents to vectorstore: {e}")
            raise
    
    def process_all_pdfs(self, pdf_folder: str, csv_path: str):
        """Process all PDFs in the folder and add to vectorstore"""
        # Setup
        self.setup_pinecone_index()
        self.load_metadata_csv(csv_path)
        
        # Get all PDF files
        pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        if not pdf_files:
            logger.warning("No PDF files found!")
            return
        
        all_documents = []
        
        # Process each PDF
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            pdf_documents = self.process_single_pdf(pdf_path)
            all_documents.extend(pdf_documents)
        
        logger.info(f"Total documents to upload: {len(all_documents)}")
        
        # Add to vectorstore
        if all_documents:
            self.add_documents_to_vectorstore(all_documents)
            logger.info("Successfully uploaded all documents to Pinecone!")
        else:
            logger.warning("No documents to upload!")
    
    def search_similar(self, query: str, k: int = 5) -> List[Document]:
        """
        Search for similar documents (useful for testing)
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Run setup_pinecone_index() first.")
        
        return self.vectorstore.similarity_search(query, k=k)


class WordBasedTextSplitter(RecursiveCharacterTextSplitter):
    """Custom text splitter that works with word counts instead of character counts"""
    
    def __init__(self, chunk_size_words: int = 300, chunk_overlap_words: int = 50, **kwargs):
        # Convert word counts to approximate character counts (assuming 5 chars per word)
        chunk_size_chars = chunk_size_words * 5
        chunk_overlap_chars = chunk_overlap_words * 5
        
        super().__init__(
            chunk_size=chunk_size_chars,
            chunk_overlap=chunk_overlap_chars,
            length_function=self._word_count_length,
            **kwargs
        )
        
        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words
    
    def _word_count_length(self, text: str) -> int:
        """Return word count instead of character count"""
        return len(text.split())


def create_word_based_processor(pinecone_api_key: str, pinecone_index_name: str = "court-judgments"):
    """
    Create a processor with word-based chunking for more precise control
    """
    processor = LangChainCourtJudgmentProcessor(pinecone_api_key, pinecone_index_name)
    
    # Replace the text splitter with word-based version
    processor.text_splitter = WordBasedTextSplitter(
        chunk_size_words=300,
        chunk_overlap_words=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    return processor


def main():
    load_dotenv()

    # Configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", '')
    PDF_FOLDER = "Sample PDFs"
    CSV_PATH = "sample_cleaned_df.csv"
    INDEX_NAME = "court-judgments"
    
    # Initialize processor (with word-based chunking for exact requirements)
    # processor = create_word_based_processor(
    #     pinecone_api_key=PINECONE_API_KEY,
    #     pinecone_index_name=INDEX_NAME
    # )
    processor = LangChainCourtJudgmentProcessor(
        pinecone_api_key=PINECONE_API_KEY,
        pinecone_index_name=INDEX_NAME
    )
    
    # Process all PDFs
    try:
        processor.process_all_pdfs(PDF_FOLDER, CSV_PATH)
        print("Processing completed successfully!")
        
        # # Example search (optional)
        # print("\n--- Testing search functionality ---")
        # results = processor.search_similar("contract dispute", k=3)
        # for i, doc in enumerate(results, 1):
        #     print(f"\nResult {i}:")
        #     print(f"Judgment ID: {doc.metadata.get('judgment_id', 'N/A')}")
        #     print(f"Pages: {doc.metadata.get('start_page', 'N/A')}-{doc.metadata.get('end_page', 'N/A')}")
        #     print(f"Case Numbers: {doc.metadata.get('case_numbers', [])}")
        #     print(f"Preview: {doc.page_content[:200]}...")
            
    except Exception as e:
        print(f"Error during processing: {e}")
        logger.error(f"Processing failed: {e}")


if __name__ == "__main__":
    main()