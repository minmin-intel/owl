from camel.toolkits import FunctionTool
import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig, CacheMode
from duckduckgo_search import DDGS
from googlesearch import search as gsearch
import os
from openai import OpenAI, AsyncOpenAI
import time


WORKDIR = os.getenv("WORKDIR")
DATAPATH= os.path.join(WORKDIR, "datasets/test_web_search/")

class DuckDuckGoSearchEngine:
    # require duckduckgo_search==6.3.5 to avoid rate limit error
    async def perform_search(self, query, num_results=3, *args, **kwargs):
        """DuckDuckGo search engine."""
        print(f"Searching DuckDuckGo for: {query}")
        results = DDGS().text(query, max_results=num_results)
        urls = [result["href"] for result in results]
        time.sleep(10)
        return urls
    
class GoogleSearchEngine:
    async def perform_search(self, query, num_results=3, *args, **kwargs):
        """Google search engine."""
        results = gsearch(query, num_results=num_results, unique=True, advanced=True)
        for res in results:
            print(res)
    
class WebCrawler:
    def __init__(self, browser_config=None, run_config=None):
        if browser_config is None:
            browser_config = BrowserConfig(verbose=True)
        if run_config is None:
            run_config = CrawlerRunConfig(
                # Content filtering
                word_count_threshold=10,
                excluded_tags=['form', 'header'],
                exclude_external_links=True,

                # Content processing
                process_iframes=True,
                remove_overlay_elements=True,

                # Cache control
                cache_mode=CacheMode.ENABLED  # Use cache if available
            )
        self.browser_config = browser_config
        self.run_config = run_config
        self.crawler = AsyncWebCrawler(config=self.browser_config)    
    async def crawl(self, url):
        async with self.crawler as crawler:
            result = await crawler.arun(
                url=url,
                config=self.run_config
            )

            if result.success:
                # # Print clean content
                # print("Content:", result.markdown[:500])  # First 500 chars
                # output_path = os.path.join(DATAPATH, "crawl4ai_content.md")
                # with open(output_path, "w") as f:
                #     f.write(result.markdown)
                return {"url":url,"content":result.markdown}

                # # Process images
                # for image in result.media["images"]:
                #     print(f"Found image: {image['src']}")

                # # Process links
                # for link in result.links["internal"]:
                #     print(f"Internal link: {link['href']}")

            else:
                print(f"Crawl failed: {result.error_message}")
                return f"Web crawl failed with url: {url}"


class LLMExtractor:
    def __init__(self, provider="together", model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo"):
        if provider == "together":
            self._api_key = os.getenv("TOGETHER_API_KEY")
            self._url = "https://api.together.xyz/v1"
        else:
            raise ValueError(f"Model provider {provider} is not supported.")
        
        self.client = OpenAI(
            timeout=180,
            max_retries=3,
            api_key=self._api_key,
            base_url=self._url,
        )
        # self._async_client = AsyncOpenAI(
        #     timeout=180,
        #     max_retries=3,
        #     api_key=self._api_key,
        #     base_url=self._url,
        # )

        self.model_name = model_name

    def run(self, webpages, query):
        # webpages: list of dicts with keys: url, content
        context = ""

        for i, webpage in enumerate(webpages):
            if isinstance(webpage, dict):
                url = webpage["url"]
                content = webpage["content"]
                breaker = "="*10
                context += f"[{i}] source: {url}\n{content}\n{breaker}\n"
            else:
                pass
                
        user_msg = """\
Below are some pages crawled from the web. Please read through them carefully and answer the question.
Web content:
{context}
===== End of web content =====
Question: {question}
"""

        # # Get response information
        if context:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": user_msg.format(context=context, question=query)}],
                max_tokens=2048,
                temperature=0.2,
                )
            resp_msg = resp.choices[0].message.content
            prompt_tokens = resp.usage.prompt_tokens
            completion_tokens = resp.usage.completion_tokens
        else:
            resp_msg = "No web content found."
            prompt_tokens = 0
            completion_tokens = 0
        return resp_msg, prompt_tokens, completion_tokens


def search_web(query: str) -> str:
    r"""
    Search the web for information on a given query.
    Args:
        query (str): Detailed search query. Only include the query, not any site info.
    Returns:
        str: web search result.
    """
    ddg = DuckDuckGoSearchEngine()
    crawler = WebCrawler()
    results = asyncio.run(ddg.perform_search(query))
    print(results)
    print(f"Found {len(results)} results.")

    webpages = []
    for url in results:
        print(f"Crawling {url}")
        web_dict = asyncio.run(crawler.crawl(url))
        if isinstance(web_dict, dict):
            webpages.append(web_dict)
    
    llm = LLMExtractor(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
    response, prompt_tokens, completion_tokens = llm.run(webpages, query)
    print(f"Search web - Response: {response}")
    print(f"Search web - Prompt tokens: {prompt_tokens}")
    print(f"Search web - Completion tokens: {completion_tokens}")
    return response

search_web_tool = FunctionTool(search_web)

if __name__ == "__main__":
    # print(search_web_tool.get_function_name())
    # print(search_web_tool.get_function_description())
    # print(search_web_tool.get_openai_function_schema())
    # # query = "Ice cream standards in the United States"
    # query = "What was the actual enrollment count of the clinical trial on H. pylori in acne vulgaris patients from Jan-May 2018 as listed on the NIH website?"
    # # response, prompt_tokens, completion_tokens = search_web(query)
    # # print(response)
    # resp = search_web_tool(query)
    # print(resp)

    # gsearchengine = GoogleSearchEngine()
    # query = "Ice cream standards in the United States"
    # results = asyncio.run(gsearchengine.perform_search(query))

    ddg = DuckDuckGoSearchEngine()
    query_list = [
        # "weather in New York",
        # "best restaurants in New York",
        # "top tourist attractions in New York",
        # "best hotels in New York",
        # "best schools in New York",
        "site:clinicaltrials.gov H. pylori acne vulgaris January 2018 May 2018"
    ]

    # sleep_time = 5
    for query in query_list:
        search_web_tool(query)
        print("="*50)
        # time.sleep(sleep_time)
    
