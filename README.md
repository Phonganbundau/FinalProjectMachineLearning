
# PDF Question Answering Application with RAG

This application provides a question-answering system for PDF and text files using a retrieval-augmented generation (RAG) approach powered by Hugging Face models and custom Chainlit utilities.

## Features

- **File Processing**: Supports text and PDF files, extracting content and generating searchable vector databases.
- **Advanced Question Answering**: Leverages a conversational retrieval chain to provide contextual answers based on the content of uploaded documents.
- **Vector Database Management**: Manages a persistent vector database for efficient query retrieval.
- **Local Language Model**: Utilizes a locally stored language model optimized for Vietnamese.

## Installation

To run this application, you need to have Python and the following packages installed:

```bash
pip install chainlit torch transformers langchain langchain_huggingface langchain_community langchain_chroma
```

Ensure that you also have the necessary model files downloaded or accessible locally.

## Usage

1. **Start the Application**:
   Run the application script in your terminal. The application will prompt you to upload a PDF or text file if no vector database exists.
   
2. **Upload a File**:
   Follow the on-screen instructions to upload a file. The application supports `.txt` and `.pdf` formats.

3. **Ask Questions**:
   Once the file is processed and the vector database is ready, you can start asking questions related to the document's content.

## Application Workflow

1. **File Upload**:
   - Users are prompted to upload a file.
   - The application processes the file and creates or updates the vector database.

2. **Question Answering**:
   - The system retrieves information from the vector database to provide answers.
   - Users can interact with the system through a chat interface to ask further questions.

3. **Session Management**:
   - The session maintains the state of the vector database and conversation history.
   - Responses are generated using a local language model, ensuring fast and relevant outputs.

### Start the app

```bash
chainlit run app.py --host 0.0.0.0 --port 8000    
```

## Contact

For any questions or issues, feel free to open an issue or reach out via email at [phonganbundau@gmail.com](mailto:phonganbundau@gmail.com) or [trantattri@gmail.com](mailto:trantattri@gmail.com)



