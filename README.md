NLP/GenAI Search System Proto-type
This repository presents a prototype of an AI-powered generative search system leveraging Retrieval-Augmented Generation (RAG). The goal is to develop a context-aware question-answering system that can directly extract relevant information from PDF documents (like the ING Health Insurance Policy Booklet). This solution is designed to enhance search efficiency and accuracy in various sectors, including legal, financial, medical, and academic fields.

Features
Generative AI-powered Search: Uses RAG pipeline to provide accurate and contextually relevant answers.
PDF Text Extraction: Extracts and processes text from PDF documents, enabling precise search queries.
ChromaDB Integration: Stores and retrieves embeddings for efficient semantic search.
Pre-trained Language Models: Utilizes Sentence Transformers for embedding generation.
Technologies Used
Python: Core language for the project.
pdfplumber: Extracts text and tables from PDF files.
Sentence-Transformers: Used to compute document embeddings.
ChromaDB: A vector database for efficient storage and retrieval of embeddings.
HuggingFace: For accessing pre-trained models like all-MiniLM-L6-v2.
Setup Instructions
Clone the Repository

bash
Copy code
git clone https://github.com/yourusername/NLP-GenAI-Search-System.git
cd NLP-GenAI-Search-System
Install Dependencies

Use the following command to install the necessary Python libraries:

bash
Copy code
pip install -U -q pdfplumber tiktoken openai chromadb sentence-transformers
Set up the Environment

Mount Google Drive (if working on Google Colab).
Ensure your Hugging Face API key is properly set in the API_Key.txt file.
Run the Notebook

Open the Jupyter notebook or Google Colab environment and execute the cells sequentially to process PDF documents and build the RAG-based question-answering system.
Data Source
Document: ING Health Insurance Policy Booklet.
Format: PDF.
Architecture
The system follows a multi-step process:

PDF Extraction: Extracts text and tables from the PDF using pdfplumber.
Text Preprocessing: Cleans and prepares extracted data for embedding.
Embedding Generation: Converts the text into embeddings using a pre-trained transformer model.
Database Storage: Stores the embeddings and document text in a ChromaDB collection.
Search Query Handling: Accepts user queries and searches for relevant information from the stored documents.
Example Usage
Extract Text from PDF:

You can extract text from the specified PDF file using the following function:

python
Copy code
extracted_text = extract_text_from_pdf(pdf_path)
Search Query:

To query the system, simply input your question:

python
Copy code
query = "What is covered?"
Results:

The system will return relevant document snippets based on the query.

Future Improvements
Improved Search Performance: Implement more advanced caching mechanisms.
Support for Multiple Document Formats: Extend the system to support various file formats like Word, HTML, etc.
Fine-Tuning Models: Fine-tune the pre-trained models for better domain-specific accuracy.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
pdfplumber: For text extraction from PDFs.
Sentence-Transformers: For embedding generation.
Hugging Face: For the pre-trained transformer models.
