from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from typing import Dict, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from langchain_mistralai.chat_models import ChatMistralAI

import chainlit as cl
import os

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

@cl.on_chat_start
async def on_chat_start():
    # app_user = cl.user_session.get("user")
    # await cl.Message(f"Hello {app_user.identifier}").send()
    
    model = ChatMistralAI()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Your name is Carlo. You are a Catholic theolgian with comprehensive knowledge of Catholic tradition, teaching, and relogious practice. You are a helpful chat assistant that helps people answer questions.",
            ),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)

    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    global DB
    DB = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    query_text = message.content

    # Search the DB.
    results = DB.similarity_search_with_relevance_scores(query_text, k=5) # k = number of tokens to retrieve
    
    if len(results) == 0 or results[0][1] < 0.3: # this needs to be tweaked. I have a feeling it's because the embedding function is kinda mid Should be around like .725
        print(f"Unable to find matching results.")
        new_message = cl.Message(content=query_text)
    else:
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        new_message = cl.Message(content=prompt)
        print(prompt)   
        source_documents = sorted([doc for doc, _score in results], key=lambda doc: doc.metadata['page'])
        source_documents = [source_documents[i] for i in range(len(source_documents)) if i == 0 or source_documents[i].metadata['page'] != source_documents[i-1].metadata['page']]
        text_elements = []  # type: List[cl.Text]
        if source_documents:
            for source_doc in source_documents:
                filename = ' '.join(source_doc.metadata['source'].split('\\')[-1].split('.')[0].split('-'))
                source_name = f"{filename} - Page: {source_doc.metadata['page']}"
                # Create the text element referenced in the message
                text_elements.append(
                    cl.Text(content=source_doc.page_content, name=source_name, display="side")
                )
            source_names = [text_el.name for text_el in text_elements]

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": new_message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    
    if source_names:
        await msg.stream_token( f"\nSources: {', '.join(source_names)}")
    else:
        await msg.stream_token( "\nNo sources found")

    await msg.send()

@cl.oauth_callback
def oauth_callback(
  provider_id: str,
  token: str,
  raw_user_data: Dict[str, str],
  default_user: cl.User,
) -> Optional[cl.User]:
  return default_user

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="What is faith?",
            message="What is faith?",
            icon="public/dove.svg",
            ),
        cl.Starter(
            label="Freedom",
            message="What sets man free?",
            icon="public/prayer.svg",
            ),
        ]