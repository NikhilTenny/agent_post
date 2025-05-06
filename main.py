from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from urllib.request import urlopen
from langchain.output_parsers.json import SimpleJsonOutputParser
import justext
from newspaper import Article
import time
import textwrap
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm_model = "gpt-3.5-turbo"


def web_search(query: str) -> list[str]:
    """
    Perform a search using the DuckDuckGo search engine.

    Parameters
    ----------
    query : str
        The search query

    Returns
    -------
    list[str]
        A list of search results
    """
    logger.info("Searching for articles...")
    try:
        wrapper = DuckDuckGoSearchAPIWrapper(region="de-de", time="d", max_results=3)
        search = DuckDuckGoSearchResults(output_format='list', wrapper=wrapper)
        search_results = search.invoke(query)
        return search_results if isinstance(search_results, list) else []
    except Exception as e:
        logger.error(f"Error during web search: {e}")
        return []




def extract_article(url: str) -> dict[str, str]:
    """
    Extract the title and text from an article.

    Parameters
    ----------
    url : str
        The URL of the article

    Returns
    -------
    dict[str, str]
        A dictionary containing the title and text of the article
    """
    logger.info("Extracting article...")
    try:
        article = Article(url)
        article.download()
        article.parse()
        return {
            "title": article.title,
            "text": article.text,
            "error": None
        }
    except Exception as e:
        logger.error(f"Error extracting article: {e}")
        return {
            "title": None,
            "text": None,
            "error": str(e)
        }

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
    query = "new discovery"
    search_results = web_search(query)
    article_url = search_results[0]['link']
    logger.info(f"Article URL: {article_url}")

    article = extract_article(article_url)
    
    s = time.perf_counter()
    input_data = {'content': article['text']}
    summary = generate_response(summarize_chain, input_data)
    elapsed = round(time.perf_counter() - s, 3)
    logger.info(f"Summary generated in {elapsed}s")
    total_tokens += summary.response_metadata['token_usage']['total_tokens']

    s = time.perf_counter()
    input_data = {'content': summary, 'title': article['title']}
    post = generate_response(post_chain, input_data)
    elapsed = round(time.perf_counter() - s, 3)
    logger.info(f"Post generated in {elapsed}s")
    total_tokens += post.response_metadata['token_usage']['total_tokens']

    logger.info(f"\n{post.content}\n")

    logger.info(f"Total tokens used: {total_tokens}") 

    
