
## Building a legal advisory system using the RAG method
The aim of this project is to develop an innovative legal advisory system that utilizes the Retrieval-Augmented Generation (RAG) approach. This technology combines the power of information retrieval with advanced natural language generation techniques to provide accurate, contextually relevant legal advice.

## Key Objectives:

Integration of Legal Databases: The system will integrate comprehensive legal databases that contain statutes, case law, and doctrinal writings. This foundational data will support the retrieval component of the RAG system, allowing it to access a wide range of legal texts and precedents.

Implementation of RAG Technology: Utilizing the RAG framework, the system will generate responses to legal queries by retrieving relevant information from its database and then using a generative model to synthesize this information into coherent, actionable advice. This method ensures that the advice is both accurate and contextually tailored to the user's specific situation.

User-Friendly Interface: The system will feature a user-friendly interface that allows users to easily input their legal questions and receive clear, understandable legal advice. The interface will be designed to accommodate both legal professionals and the general public, making legal advice more accessible to a broader audience.

## Features

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
   
2. **Ask Questions**:
   Once the file is processed and the vector database is ready, you can start asking questions related to the document's content.

## Application Workflow


1. **Question Answering**:
   - The system retrieves information from the vector database to provide answers.
   - Users can interact with the system through a chat interface to ask further questions.

2. **Session Management**:
   - The session maintains the state of the vector database and conversation history.
   - Responses are generated using a local language model, ensuring fast and relevant outputs.

### Start the app

```bash
chainlit run app.py --host 0.0.0.0 --port 8000    
```

## Contact

For any questions or issues, feel free to open an issue or reach out via email at [phonganbundau@gmail.com](mailto:phonganbundau@gmail.com) or [trantattri@gmail.com](mailto:trantattri@gmail.com)



