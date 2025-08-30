#use google search and selenium to fetch dynamic pages
# =====================
# summary_lmstudio_fullpage.py
# =====================
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
import re
import time

# ---------------------
# تعریف State
# ---------------------
class State(TypedDict):
    messages: list
    need_search: str

# ---------------------
# اتصال به LM Studio
# ---------------------
llm = ChatOpenAI(
    model="gemma-3-4b-it@q4_k_m",
    openai_api_base="http://localhost:4050/v1",
    openai_api_key="lmstudio",
    temperature=0.1
)

# ---------------------
# ---------- توابع کمکی ----------
def _selenium_fetch_single(url: str, max_chars: int = 4000) -> str:
    """
    یک URL را با Selenium headless باز کرده و متن تمیز را برمی‌گرداند.
    """
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.common.exceptions import WebDriverException
    from bs4 import BeautifulSoup
    from webdriver_manager.chrome import ChromeDriverManager
    import re

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

        # اسکرول تا انتها برای لود کامل
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


def _google_search(query: str, num: int = 3) -> list[str]:
    """
    جستجوی سریع در گوگل و بازگرداندن لینک‌ها
    """
    try:
        from googlesearch import search as googlesearch_search
        return list(googlesearch_search(query, num_results=num, lang="en"))
    except Exception:
        return []

# ---------------------
# ---------- Nodes ----------
def check_need_search(state: State):
    user_msg = state["messages"][-1].content
    response = llm.invoke([
        HumanMessage(content=f"آیا برای پاسخ به موضوع زیر نیاز به جستجو در اینترنت هست؟ فقط جواب 'بله' یا 'خیر'.\n\nموضوع: {user_msg}")
    ])
    return {"need_search": "بله" if "بله" in response.content else "خیر"}


def search(state: State):
    """
    ورودی/خروجی کاملاً مشابه نود قبلی
    - عبارت بهینه می‌سازد
    - در گوگل سرچ می‌کند
    - با Selenium صفحات داینامیک را می‌خواند
    - متن یکپارچه را به LLM می‌سپارد
    """
    user_msg = state["messages"][-1].content

    # 1) عبارت جستجوی بهینه
    optimized_query_msg = HumanMessage(
        content=f"برای پاسخ به این سوال، تنها چند کلمهٔ کلیدی که بیشترین احتمال نتیجه در گوگل را دارند بنویس، بدون توضیح اضافه:\n{user_msg}"
    )
    optimized_query = llm.invoke([optimized_query_msg]).content.strip().splitlines()[0]
    print(f"🔍 عبارت جستجوی گوگل: {optimized_query}")

    # 2) جستجوی گوگل
    links = _google_search(optimized_query, 3)

    # 3) استخراج متن با Selenium
    full_texts = []
    for url in links:
        print(f"⬇️  Selenium در حال دانلود: {url}")
        full_texts.append(_selenium_fetch_single(url, max_chars=4000))

    combined_text = "\n\n".join(full_texts).strip()
    if not combined_text:
        combined_text = f"هیچ اطلاعاتی از اینترنت برای '{user_msg}' پیدا نشد."

    final_msg = HumanMessage(
        content=f"با استفاده از اطلاعات زیر، یک متن یکپارچه و منسجم برای تولید جواب برای سوال کاربر که برابر با: **{user_msg}** هست بنویس (کمترین حالت و در صورت نیاز حداکثر ۳-۵ خط). مطمعن شو که جواب سوال کاربر در خلاصه تولید شده وجود دارد.:\n\n{combined_text}"
    )
    final_response = llm.invoke([final_msg])
    return {"messages": state["messages"] + [final_response]}


def llm_only(state: State):
    user_msg = state["messages"][-1].content
    response = llm.invoke([HumanMessage(content=f"موضوع زیر را توضیح بده:\n\n{user_msg}")])
    return {"messages": state["messages"] + [response]}


def summarize(state: State):
    all_msgs = "\n".join([m.content for m in state["messages"][1:]])
    response = llm.invoke([HumanMessage(content=f"مطالب زیر را خلاصه کن (کمترین حالت و در صورت نیاز حداکثر ۳-۵ خط):\n\n{all_msgs}")])
    return {"messages": state["messages"] + [response]}

# ---------------------
# ساخت گراف
# ---------------------
graph_builder = StateGraph(State)
graph_builder.add_node("check_need_search", check_need_search)
graph_builder.add_node("search", search)
graph_builder.add_node("llm_only", llm_only)
graph_builder.add_node("summarize", summarize)

graph_builder.set_entry_point("check_need_search")
graph_builder.add_conditional_edges(
    "check_need_search",
    lambda state: "search" if state["need_search"] == "بله" else "llm_only",
    {"search": "search", "llm_only": "llm_only"}
)
graph_builder.add_edge("search", "summarize")
graph_builder.add_edge("llm_only", END)
graph_builder.add_edge("summarize", END)

graph = graph_builder.compile()

# ---------------------
# اجرای برنامه
# ---------------------
if __name__ == "__main__":
    topic = input("موضوع را وارد کنید: ")
    result = graph.invoke({"messages": [HumanMessage(content=topic)], "need_search": ""})
    print("\n📌 خروجی نهایی:\n")
    for m in result["messages"]:
        print(m.content)