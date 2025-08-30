# =========================================================
# summary_lmstudio_multithread.py
# فقط DuckDuckGo + Selenium + ThreadPoolExecutor
# =========================================================
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

import re
import time
import functools
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- Selenium ----------
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager

# ---------- DuckDuckGo ----------
from ddgs import DDGS        # pip install ddgs

# =========================================================
# State
# =========================================================
class State(TypedDict):
    messages: list
    need_search: str

# =========================================================
# اتصال به LM Studio
# =========================================================
llm = ChatOpenAI(
    model="gemma-3-4b-it@q4_k_m",
    openai_api_base="http://localhost:4050/v1",
    openai_api_key="lmstudio",
    temperature=0.1
)

# =========================================================
# توابع کمکی
# =========================================================
def _selenium_fetch_single(url: str, max_chars: int = 4000) -> str:
    """یک URL را با Selenium headless باز کرده و متن تمیز را برمی‌گرداند."""
    chrome_opts = Options()
    chrome_opts.add_argument("--headless=new")
    chrome_opts.add_argument("--disable-gpu")
    chrome_opts.add_argument("--no-sandbox")
    chrome_opts.add_argument("--window-size=1920,3000")
    chrome_opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125 Safari/537.36"
    )

    driver = None
    try:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_opts
        )
        driver.get(url)

        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.2)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        html = driver.page_source
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "aside", "header", "noscript"]):
            tag.extract()

        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text)[:max_chars]
        return f"منبع: {url}\n{text}"

    except WebDriverException as e:
        return f"[خطا در Selenium برای {url}: {e}]"
    finally:
        if driver:
            driver.quit()


def _search_links(query: str, num: int = 3) -> list[str]:
    """فقط از DuckDuckGo لینک‌ها را می‌گیرد."""
    try:
        return [item["href"] for item in DDGS().text(query, region='ir', max_results=num)]
    except Exception as e:
        print(f"⚠️ DDGS خطا: {e}")
        return []

# =========================================================
# Nodes
# =========================================================
def check_need_search(state: State):
    user_msg = state["messages"][-1].content
    response = llm.invoke([
        HumanMessage(
            content=f"آیا برای پاسخ به موضوع زیر نیاز به جستجو در اینترنت هست؟ "
                    f"فقط جواب 'بله' یا 'خیر'.\n\nموضوع: {user_msg}"
        )
    ])
    return {"need_search": "بله" if "بله" in response.content else "خیر"}


def search(state: State):
    """
    1) عبارت بهینه می‌سازد
    2) فقط از DuckDuckGo لینک می‌گیرد
    3) لینک‌ها را multi-thread دانلود می‌کند
    """
    user_msg = state["messages"][-1].content

    # 1) عبارت بهینه
    optimized_query_msg = HumanMessage(
        content=f"برای پاسخ به این سوال، تنها چند کلمهٔ کلیدی که بیشترین احتمال نتیجه را دارند بنویس، "
                f"بدون توضیح اضافه:\n{user_msg}"
    )
    optimized_query = llm.invoke([optimized_query_msg]).content.strip().splitlines()[0]
    print(f"🔍 عبارت جستجوی DuckDuckGo: {optimized_query}")

    # 2) لینک‌ها از DuckDuckGo
    links = _search_links(optimized_query, 3)

    # 3) دانلود موازی
    full_texts = []
    if links:
        fetch = functools.partial(_selenium_fetch_single, max_chars=4000)
        with ThreadPoolExecutor(max_workers=min(3, len(links))) as executor:
            future_to_url = {executor.submit(fetch, url): url for url in links}
            for i, future in enumerate(as_completed(future_to_url), 1):
                url = future_to_url[future]
                try:
                    text = future.result()
                    full_texts.append(text)
                    print(f"[{i}/{len(links)}] دانلود شد: {url}")
                except Exception as e:
                    full_texts.append(f"[خطا در دانلود {url}: {e}]")

    combined_text = "\n\n".join(full_texts).strip()
    if not combined_text:
        combined_text = f"هیچ اطلاعاتی از اینترنت برای '{user_msg}' پیدا نشد."

    # 4) تولید پاسخ نهایی
    final_msg = HumanMessage(
        content=f" مطمعن شو که جواب سوال کاربر در خلاصه تولید شده وجود دار با استفاده از اطلاعات زیر، یک متن یکپارچه و منسجم بنویس "
                f"(کمترین حالت و در صورت نیاز حداکثر ۳-۵ خط):\n\n{combined_text}"
    )
    final_response = llm.invoke([final_msg])
    return {"messages": state["messages"] + [final_response]}


def llm_only(state: State):
    user_msg = state["messages"][-1].content
    response = llm.invoke([HumanMessage(content=f"موضوع زیر را توضیح بده:\n\n{user_msg}")])
    return {"messages": state["messages"] + [response]}

# =========================================================
# ساخت گراف
# =========================================================
graph_builder = StateGraph(State)
graph_builder.add_node("check_need_search", check_need_search)
graph_builder.add_node("search", search)
graph_builder.add_node("llm_only", llm_only)

graph_builder.set_entry_point("check_need_search")
graph_builder.add_conditional_edges(
    "check_need_search",
    lambda state: "search" if state["need_search"] == "بله" else "llm_only",
    {"search": "search", "llm_only": "llm_only"}
)
graph_builder.add_edge("search", END)
graph_builder.add_edge("llm_only", END)

graph = graph_builder.compile()

# =========================================================
# اجرای برنامه
# =========================================================
if __name__ == "__main__":
    topic = input("موضوع را وارد کنید: ")
    result = graph.invoke({"messages": [HumanMessage(content=topic)], "need_search": ""})
    print("\n📌 خروجی نهایی:\n")
    for m in result["messages"]:
        print(m.content)