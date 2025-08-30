# summary_lmstudio_fullpage.py
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
import re

# =====================
# تعریف State
# =====================
class State(TypedDict):
    messages: list
    need_search: str

# =====================
# اتصال به LM Studio
# =====================
llm = ChatOpenAI(
    model="gemma-3-4b-it@q4_k_m",
    openai_api_base="http://localhost:4050/v1",
    openai_api_key="lmstudio",
    temperature=0.1
)

# ---------- توابع کمکی ----------
def fetch_text(url: str, max_chars: int = 3000) -> str:
    """
    یک URL را گرفته و متن قابل‌خواندن آن را برمی‌گرداند.
    اگر خطا رخ دهد، پیام خطا برمی‌گردد.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=8)
        r.raise_for_status()
    except Exception as e:
        return f"[خطا در دانلود {url}: {e}]"

    soup = BeautifulSoup(r.text, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "aside"]):
        tag.extract()
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text[:max_chars]

# ---------- Nodes ----------
def check_need_search(state: State):
    user_msg = state["messages"][-1].content
    response = llm.invoke([
        HumanMessage(content=f"آیا برای پاسخ به موضوع زیر نیاز به جستجو در اینترنت هست؟ فقط جواب 'بله' یا 'خیر'.\n\nموضوع: {user_msg}")
    ])
    return {"need_search": "بله" if "بله" in response.content else "خیر"}

def search(state: State):
    user_msg = state["messages"][-1].content

    # عبارت جستجوی کوتاه
    optimized_query_msg = HumanMessage(
        content=f"برای پاسخ به این سوال، تنها چند کلمهٔ کلیدی که بیشترین احتمال دارد در اینترنت نتیجه بدهد را بنویس، بدون توضیح اضافه:\n{user_msg}"
    )
    optimized_query = llm.invoke([optimized_query_msg]).content.strip().splitlines()[0]
    print(f"🔍 عبارت جستجوی بهینه‌شده: {optimized_query}")

    # گرفتن لینک‌ها از DuckDuckGo
    links = []
    try:
        for item in DDGS().text(optimized_query, max_results=3):
            links.append(item["href"])
    except Exception as e:
        links = []

    # دانلود و استخراج متن هر لینک
    full_texts = []
    for url in links:
        print(f"⬇️  در حال دانلود: {url}")
        page_text = fetch_text(url)
        full_texts.append(f"منبع: {url}\n{page_text}\n")

    combined_text = "\n".join(full_texts).strip()
    print("📄 متن کامل صفحات:\n" + (combined_text or "نتیجه‌ای پیدا نشد.") + "\n")

    if not combined_text:
        combined_text = f"هیچ اطلاعاتی از اینترنت برای '{user_msg}' پیدا نشد."

    final_msg = HumanMessage(
        content=f"با استفاده از اطلاعات زیر، یک متن یکپارچه و منسجم بنویس (کمترین حالت و در صورت نیاز حداکثر ۳-۵ خط):\n\n{combined_text}"
    )
    final_response = llm.invoke([final_msg])
    return {"messages": state["messages"] + [HumanMessage(content=combined_text)]
    }

def filter_relevant_paragraphs(state: State):
    """
    متن‌های دریافتی از وب را به پاراگراف‌ها تقسیم کرده
    و فقط پاراگراف‌های مرتبط با موضوع اصلی را حفظ می‌کند.
    """
    user_msg = state["messages"][0].content          # پیام اولیه کاربر
    raw_text = state["messages"][-1].content
    combined_text = state["messages"][-1].content    # متن خام دانلود‌شده

    paragraphs = [p.strip() for p in combined_text.split("\n") if p.strip()]

    # ارسال دسته‌ای برای صرفه‌جویی در توکن
    batch_size = 10
    relevant_chunks = []

    for i in range(0, len(paragraphs), batch_size):
        batch = "\n".join(paragraphs[i:i+batch_size])
        prompt = (
            f"موضوع اصلی: {user_msg}\n\n"
            f"پاراگراف‌های زیر را بررسی کن. فقط پاراگراف‌هایی که به موضوع اصلی مرتبط‌اند "
            f"را بدون تغییر برگردان؛ بقیه را حذف کن. "
            f"اگر هیچ‌کدام مرتبط نبودند، بنویس: «هیچ پاراگراف مرتبطی یافت نشد».\n\n{batch}"
        )
        response = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        if "هیچ پاراگراف مرتبطی یافت نشد" not in response:
            relevant_chunks.append(response)

    filtered_text = "\n".join(relevant_chunks) or "اطلاعات مرتبطی یافت نشد."
    return {"messages": state["messages"][:-1] + [HumanMessage(content=filtered_text)]}

def llm_only(state: State):
    user_msg = state["messages"][-1].content
    response = llm.invoke([HumanMessage(content=f"موضوع زیر را توضیح بده:\n\n{user_msg}")])
    return {"messages": state["messages"] + [response]}

def summarize(state: State):
    all_msgs = "\n".join([m.content for m in state["messages"][1:]])
    response = llm.invoke([HumanMessage(content=f"مطالب زیر را خلاصه کن (کمترین حالت و در صورت نیاز حداکثر ۳-۵ خط):\n\n{all_msgs}")])
    return {"messages": state["messages"] + [response]}

# =====================
# ساخت گراف
# =====================
graph_builder = StateGraph(State)
graph_builder.add_node("check_need_search", check_need_search)
graph_builder.add_node("search", search)
graph_builder.add_node("llm_only", llm_only)
graph_builder.add_node("summarize", summarize)
graph_builder.add_node("filter_relevant_paragraphs", filter_relevant_paragraphs)

graph_builder.set_entry_point("check_need_search")
graph_builder.add_conditional_edges(
    "check_need_search",
    lambda state: "search" if state["need_search"] == "بله" else "llm_only",
    {"search": "search", "llm_only": "llm_only"}
)
graph_builder.add_edge("search", "filter_relevant_paragraphs")
graph_builder.add_edge("filter_relevant_paragraphs", "summarize")
graph_builder.add_edge("llm_only", END)
graph_builder.add_edge("summarize", END)

graph = graph_builder.compile()

# =====================
# اجرای برنامه
# =====================
if __name__ == "__main__":
    topic = input("موضوع را وارد کنید: ")
    result = graph.invoke({"messages": [HumanMessage(content=topic)], "need_search": ""})
    print("\n📌 خروجی نهایی:\n")
    for m in result["messages"]:
        print(m.content)