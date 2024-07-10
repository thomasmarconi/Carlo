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

def sort_sources(sources):
    return list({source: None for source in [doc.metadata.get("source", None) for doc, _score in sorted(sources, key=lambda x: x[1], reverse=True)]}.keys())

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---
History: {history}

Answer the question based on the above context and history: {question}
"""

GENERAL_PROMPT_TEMPLATE="""
History: {history}

Answer the question with the above history as additional context: {question}
"""

EMBEDDING_FUNCTION = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
DB = Chroma(persist_directory=CHROMA_PATH, embedding_function=EMBEDDING_FUNCTION)

@cl.on_chat_start
async def on_chat_start():
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

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable
    query_text = message.content

    sources=None
    pdfs=None

    # Search the DB.
    results = DB.similarity_search_with_relevance_scores(query_text, k=5) # k = number of tokens to retrieve
    
    if len(results) == 0 or results[0][1] < 0.3: # this needs to be tweaked. I have a feeling it's because the embedding function is kinda mid Should be around like .725 Well actually it looks like the values are negative so I might be interpreting the results wrong
        print(f"Unable to find matching results.")
        prompt_template = ChatPromptTemplate.from_template(GENERAL_PROMPT_TEMPLATE)
        prompt = prompt_template.format(
            history=cl.chat_context.to_openai(),
            question=query_text)
    else:
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        sources = sort_sources(results)
        pdfs = [cl.Pdf(path=source, name=os.path.basename(source), display="side") for source in sources]
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(
            history=cl.chat_context.to_openai(),
            context=context_text,
            question=query_text)

    new_message = cl.Message(content=prompt)
    msg = cl.Message(content="", elements=pdfs)

    async for chunk in runnable.astream(
        {"question": new_message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    if sources:
        # print(sources)
        await msg.stream_token(f"\nSources: {', '.join([os.path.basename(source) for source in sources])}" )

    await msg.send()

@cl.oauth_callback
def oauth_callback(provider_id: str, token: str, raw_user_data: Dict[str, str], default_user: cl.User) -> Optional[cl.User]:
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