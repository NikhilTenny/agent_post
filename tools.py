from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from newspaper import Article
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import validators
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    logger.info(f"Extracting article: {url}")
    try:
        url = url.strip("'")
        validation = validators.url(url)
        if not validation:
            return {
                "title": None,
                "text": None,
                "error": "Invalid URL"
            }
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


summarize_prompt = PromptTemplate(
    input_variables=["content"],
    template=summarize_template,
)

llm_model = "gpt-3.5-turbo"
llm = ChatOpenAI(model=llm_model, temperature=0)

summarize_chain = summarize_prompt | llm

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
    logger.info(f"Summarizing text...{text}")
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

    Content:
    {content}

"""
post_prompt = PromptTemplate(
    input_variables=["content"],
    template=post_template,
)
post_chain = post_prompt | llm

def generate_insta_post(input_data: str) -> str:
    """
    Generate an Instagram post using the given summarized text.

    Parameters
    ----------
    input_data : str
        The summarized text to use for generating the response.

    Returns
    -------
    str
        The generated Instagram post; returns an empty string on failure.
    """
    logger.info("Generating Instagram post...")
    try:
        response = post_chain.invoke({'content': input_data})
        return response.content
    except Exception as e:
        logger.error(f"Error during Instagram post generation: {e}")
        return ""


def write_to_file(text: str) -> bool:
    """
    Write the given text to a file.

    Parameters
    ----------
    text : str
        The text to write to the file.

    Returns
    -------
    bool
        True if the write was successful, False otherwise.
    """
    try:
        with open('article.txt', 'a') as f:
            f.write(text)
        logger.info(f"Successfully wrote to article.txt")
        return True
    except Exception as e:
        logger.error(f"Error writing to file article.txt: {e}")
        return False

def send_email(email_body: str):
    # Email credentials
    sender_email = os.getenv("SENDER_EMAIL")
    app_password = os.getenv("APP_PASSWORD")
    receiver_email = os.getenv("RECEIVER_EMAIL")

    # Create message
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "Email from Agentic Pundachi"

    # Attach body to email
    message.attach(MIMEText(email_body, "plain"))

    # Send the email
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, app_password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        logger.info("Email sent successfully!")
    except Exception as e:
        logger.error("Error:", e)
