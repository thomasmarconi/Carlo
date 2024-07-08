import argparse
import os
from dotenv import load_dotenv
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace, HuggingFacePipeline
from huggingface_hub import InferenceClient
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_mistralai.chat_models import ChatMistralAI

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=5) # k = number of tokens to retrieve
    if len(results) == 0: # or results[0][1] < 0.7: # this needs to be messed with
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    # Using the HuggingFace Inference Client
    # client = InferenceClient(
    #     "mistralai/Mixtral-8x7B-Instruct-v0.1",
    #     token=os.environ['HUGGINGFACEHUB_API_TOKEN'],
    # )
    # response = client.chat_completion(
    #     messages=[{"role": "user", "content": prompt}],
    #     max_tokens=1000,
    #     stream=False,
    # )
    # response_text = response.choices[0].message.content

    # Using the HuggingFaceEndpoint Technique
    # llm = HuggingFaceEndpoint(
    #     repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    #     task="text-generation",
    #     max_new_tokens=1024,
    #     do_sample=False,
    #     top_k= 50,
    #     temperature= 0.1,
    #     repetition_penalty=1.03,
    # )
    # messages = [
    #     SystemMessage(content="You're a knowledgeable Catholic theologian who has comprehensive knowledge of Catholic tradition, teaching, and practice."),
    #     HumanMessage(
    #         content=prompt
    #     ),
    # ]
    # chat_model = ChatHuggingFace(llm=llm)
    # response_text = chat_model.invoke(messages)

    # Using Ollamma
    # llm = ChatOllama(model="llama3")
    # ollama_prompt = ChatPromptTemplate.from_template("{temp}")
    # chain = ollama_prompt | llm | StrOutputParser()
    # response_text = chain.invoke({"temp": prompt})

    # Using Mistral API
    chat = ChatMistralAI()
    response_text = chat.invoke([HumanMessage(content=prompt)]).content

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()
