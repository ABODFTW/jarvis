# from langchain_openai import ChatOpenAI # if you want to use openai

from langchain_ollama import ChatOllama

# Initialize LLM
llm = ChatOllama(model="qwen3:1.7b", reasoning=False)