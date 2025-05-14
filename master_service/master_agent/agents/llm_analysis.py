from dotenv import load_dotenv
import os
import requests


class AnalysisPrompt:
    def __init__(self, current_price, sources, ticker) -> None:
        self.ticker = ticker
        self.current_price = current_price
        self.sources = sources
        self.url_site = []
        self.formatted_sources = self._format_sources()
        self.prompt = self._build_prompt()

    def _format_sources(self):
        formatted = ""
        for i, source in enumerate(self.sources):
            self.url_site.append(source['url'])
            formatted += f"Source {i}: {source['url']}\nContent: {source['content']}\n\n"
        return formatted

    def _build_prompt(self):
        return f"""
You are a financial analyst specializing in stock market analysis. You are tasked with summarizing and analyzing a specific stock and providing a sentiment analysis based on given sources. 
Cite the source after the sentence in markdown format. eg. [source](url). Make sure the generated response is in markdown format so double check all special tokens.

Stock Symbol: {self.ticker}
URL to source sites: {self.url_site}
{self.formatted_sources}
Current Price is {self.current_price}

**Example Format Response:**
---
## Stock Summary and Analysis:

- **Current Price:** \\${self.current_price}
- **Company Overview:** XYZ is a leading provider of [products/services], operating primarily in [industry/sector]. Recently, the company has [recent significant events]. [source]({{url}})
- **Financial Performance:** The company reported a revenue of \\$X million in the last quarter, with a profit margin of Y%. EPS was \\$Z. [source]({{url}})
- **Market Trends:** The [industry/sector] is currently experiencing [describe trends]. Regulatory factors such as [example] may affect the outlook. [source]({{url}})
- **Analyst Ratings:** Analysts have given the stock a rating of [rating], with a price target range of \\$[low] to \\$[high]. [source]({{url}})

**Sentiment Analysis:**

- **Positive Sentiments:** [Summary of positive sentiments]
- **Negative Sentiments:** [Summary of negative sentiments]
- **Neutral Sentiments:** [Summary of neutral sentiments]

**Conclusion:**

- **Overall Sentiment:** The overall sentiment towards XYZ is [positive/negative/neutral].
- **Investment Recommendation:** Based on the analysis, it is recommended to [buy/hold/sell] the stock. This recommendation is supported by [key insights].

> Note: The above analysis is based on publicly available information and should not be taken as personalized investment advice.
        """

    def __str__(self):
        return self.prompt


# Load .env for LLM endpoint
load_dotenv()
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL")


class AnalysisAgent:
    def __init__(self) -> None:
        if not LLM_SERVICE_URL:
            raise ValueError("LLM_SERVICE_URL environment variable is not set.")
    
    def get_analysis(self, ticker, sources, current_price):
        prompt = AnalysisPrompt(
            ticker=ticker,
            current_price=current_price,
            sources=sources
        )
        response = requests.post(
            LLM_SERVICE_URL + "/generate",
            json={"content": str(prompt)}
        )
        response.raise_for_status()
        return response.json().get("content", "❌ No content returned")

    def run(self, data: dict):
        analysis = self.get_analysis(
            ticker=data["ticker"],
            sources=data["sources"],
            current_price=data["price"]
        )
        data["analysis"] = analysis
        return data