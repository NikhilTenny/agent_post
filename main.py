from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from urllib.request import urlopen
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from tools import web_search, extract_article, summarize_text, write_to_file, generate_insta_post, send_email


import time
import logging
import os
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm_model = os.getenv("LLM_MODEL")
temperature = os.getenv("TEMPERATURE")

memory = ConversationBufferWindowMemory(k=3, return_messages=True)
llm = ChatOpenAI(model=llm_model, temperature=temperature)

if __name__ == "__main__":
    tools = [
        Tool(
            name="web_search",
            func=web_search,
            description="Useful for when you need to answer questions about current events."
        ),
        Tool(
            name="extract_article",
            func=extract_article,
            description="Useful for when you need to extract the title and text from an article."
        ),
        Tool(
            name="summarize_text",
            func=summarize_text,
            description="Useful for when you need to summarize a long text."
        ),
        Tool(
            name="write_to_file",
            func=write_to_file,
            description="Useful for when you need to write text to a file."
        ),
        Tool(
            name="generate_insta_post",
            func=generate_insta_post,
            description="Useful for when you need to generate a instagram post using a chain and input data."
        ),
        Tool(
            name="send_email",
            func=send_email,
            description="Useful for when you need to send an email."
        )
    ]
    
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory
    )
    query = "latest Automobile discovery"
    final_prompt = """
        You are an AI assistant with access to the web, summarization tools, and content generation capabilities.
        Given a user query, follow these steps:

        Search the web and find a recent and relevant news article based on the query: {query}.

        Select one article and extract its full content.

        Summarize the article concisely in 3–5 sentences, maintaining key facts and tone.

        Based on the summary, generate a compelling Instagram post, including a short caption (under 2200 characters), an engaging hook, and relevant hashtags.

        Email the generated Instagram post.

        Be accurate, brief, and maintain journalistic integrity when summarizing.
    """
    print(agent.run(final_prompt.format(query=query)))


        










# Converting the agent to a fully autonomous agent
# For that first a proper goal is need to be defined
# Goal: Generate an Instagram post of the latest trending article
#   It should be send as an email

# Things that make up an autonomous agent
# 1. Goal
# 2. Planning
# 3. Action
# 4. Feedback

# 1.GOAL
# Find the article, get the content, summarize it and generate an instagram post and send it as an email
# -Final goal is send email

# 2. PLANNING
# Available tools are:
# -web_search
# -extract_article
# -summarize_text
# -write_to_file
# -generate_insta_post
# -send_email

