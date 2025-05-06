from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from urllib.request import urlopen
from langchain.output_parsers.json import SimpleJsonOutputParser

from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from tools import web_search, extract_article, summarize_text, write_to_file, generate_insta_post


import time
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm_model = "gpt-3.5-turbo"








content_category = "Article"
max_words = 60
entity_range = 3
iterations = 2

json_parser = SimpleJsonOutputParser()
summarize_template = """
    You are an expert summarizer. Given a long text, your task is to generate a clear and concise summary that captures all the key points and important details.

    Instructions:
    - Summarize the following content in a professional, neutral tone.
    - Focus on the main ideas and significant details.
    - Rephrase and simplify complex language when needed.
    - Do not include personal opinions or unnecessary examples.
    - The summary should be within 3-5 paragraphs.
    - If required, use bullet points to highlight key points.
    - The summary shouldn't exceed 500 words.
    - The summary should be easy to understand.

    Content:
    {content}
"""

post_template = """
    You are a social media content writer for a popular Instagram account that shares the latest news and insights in a clear, engaging format.

    Your task:
    - Read the following content and create a **short Instagram-style post** summarizing the key points.
    - Use a **clear, attention-grabbing opening line**.
    - Keep the tone **informative, modern, and neutral**, suitable for a broad audience.
    - Use **short paragraphs** or even **bullet points** for readability.
    - The total length should be **under 500 words**.
    - Avoid technical jargon; make it understandable for the average reader.
    - End with an optional CTA (e.g., “Follow for more updates like this.”)
    - Use emojis at minimal and make the post engaging.

    The actual heading of the article is:
    {title}
    Content:
    {content}

"""

summarize_prompt = PromptTemplate(
    input_variables=["content"],
    template=summarize_template,
)

post_prompt = PromptTemplate(
    input_variables=["content", "title"],
    template=post_template,
)
llm = ChatOpenAI(model=llm_model, temperature=0)

# Build chain
summarize_chain = summarize_prompt | llm
post_chain = post_prompt | llm

def summarize_text(text: str) -> str:
    """
    Summarize the given text.

    Parameters
    ----------
    text : str
        The text to summarize

    Returns
    -------
    str
        The summarized text; returns an empty string on failure.
    """
    logger.info("Summarizing text...")
    try:
        start = time.perf_counter()
        summary = summarize_chain.invoke({'content': text})
        elapsed = round(time.perf_counter() - start, 3)
        logger.info(f"Summary generated in {elapsed}s")
        logger.info(f"Total tokens used: {summary.response_metadata['token_usage']['total_tokens']}")
        return summary.content
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        return ""


def generate_response(chain, input_data: dict[str, str]) -> str:
    """
    Generate a response using the given chain and input data.

    Parameters
    ----------
    chain : LLMChain
        The chain to use for generating the response
    input_data : dict[str, str]
        The input data to use for generating the response

    Returns
    -------
    str
        The generated response
    """
    logger.info("Generating response...")
    response = chain.invoke(input_data)
    return response


total_tokens = 0
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
        )
    ]
    
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    query = "latest AI discovery"
    print(agent.run(f"Find some news article about {query}, extract the any one article, create a summary of the article, generate an instagram post based on the summary and write it to a file."))
    url = 'https:/www.popularmechanics.com/technology/robots/a60806576/new-ai-discovery/'
    # url = "https://www.popularmechanics.com/technology/robots/a60806576/new-ai-discovery/"
    # article = extract_article(url)
    # print(article)

        
