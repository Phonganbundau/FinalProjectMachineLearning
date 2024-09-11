import chainlit as cl
import torch
import os
from chainlit.types import AskFileResponse
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

PERSIST_DIRECTORY = "vectorstore/"  # Thư mục để lưu trữ vector database
model_name = "ricepaper/vi-gemma-2b-RAG"
embedding_model_name = "hiieu/halong_embedding"


text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)

def process_file(file: AskFileResponse):
    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader

    loader = Loader(file.path)
    documents = loader.load()
    docs = text_splitter.split_documents(documents)
    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{i}"
    return docs

def create_default_vector_db():
    default_texts = ["This is a default document."]  # Dữ liệu mẫu
    metadatas = [{"source": "default"}]
    return Chroma.from_texts(
        default_texts,
        embedding,
        metadatas=metadatas,
        collection_name="default_collection",
        persist_directory=PERSIST_DIRECTORY
    )

def load_or_create_vector_db(file: AskFileResponse = None):
    if os.path.exists(PERSIST_DIRECTORY):
        # Tải vector database từ thư mục
        vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)
    elif file is not None:
        # Nếu không có vector database và có tệp, tạo vector database từ tệp
        docs = process_file(file)
        vector_db = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=PERSIST_DIRECTORY)

    else:
        # Nếu không có vector database và không có tệp, tạo vector database mặc định
        vector_db = create_default_vector_db()
    
    return vector_db




def get_huggingface_llm(model_name: str, max_new_token: int = 512):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    # Sử dụng mô hình và tokenizer từ đường dẫn cục bộ
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, 
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_token,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto"
    )

    llm = HuggingFacePipeline(
        pipeline=model_pipeline,
    )
    return llm

# Khởi tạo LLM sử dụng mô hình cục bộ
LLM = get_huggingface_llm(model_name)



welcome_message = """Welcome to the PDF QA! To get started:
1. Upload a PDF or text file
2. Ask a question about the file
"""
@cl.on_chat_start
async def on_chat_start():
    # Kiểm tra xem có sẵn cơ sở dữ liệu vector không
    if os.path.exists(PERSIST_DIRECTORY):
        # Nếu đã có cơ sở dữ liệu vector, load nó và không cần yêu cầu tải file
        vector_db = await cl.make_async(load_or_create_vector_db)()

        msg = cl.Message(content="Vector database đã sẵn sàng. Bạn có thể bắt đầu đặt câu hỏi!")
        await msg.send()
    else:
        # Nếu chưa có cơ sở dữ liệu vector, yêu cầu người dùng tải file
        files = None
        while files is None:
            files = await cl.AskFileMessage(
                content=welcome_message,
                accept=["text/plain", "application/pdf"],
                max_size_mb=20,
                timeout=180,
            ).send()
        
        file = files[0]

        msg = cl.Message(content=f"Processing `{file.name}`...")
        await msg.send()

        vector_db = await cl.make_async(load_or_create_vector_db)(file)

        msg.content = f"Processing `{file.name}` done. You can now ask questions!"
        await msg.update()

    # Tạo chain cho việc xử lý truy vấn của người dùng
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={'k': 3})

    chain = ConversationalRetrievalChain.from_llm(
        llm=LLM,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

    cl.user_session.set("chain", chain)


@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")
    if chain is None:
        await cl.Message(content="Error: Chain not initialized.").send()
        return

    cb = cl.AsyncLangchainCallbackHandler()
    try:
        res = await chain.ainvoke(message.content, callbacks=[cb])
        print("Response received:", res)  # Kiểm tra phản hồi

        answer = res["answer"]
        source_documents = res["source_documents"]  

        text_elements = []  

        if source_documents:
            for source_idx, source_doc in enumerate(source_documents):
                source_name = f"source_{source_idx}"
                text_elements.append(
                    cl.Text(content=source_doc.page_content, name=source_name)
                )
            source_names = [text_el.name for text_el in text_elements]

            if source_names:
                answer += f"\nSources: {', '.join(source_names)}"
            else:
                answer += "\nNo sources found"
        else:
            answer += "\nNo sources found"

        await cl.Message(content=answer, elements=text_elements).send()

    except Exception as e:
        print("Error during invocation:", e)
        await cl.Message(content="An error occurred while processing your request.").send()


