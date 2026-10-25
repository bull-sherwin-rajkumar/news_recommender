import logging
import sqlite3
from datetime import date

import streamlit as st
from chromadb.errors import ChromaError

from news_recommender.chat import Chat
from news_recommender.config import Settings
from news_recommender.retrieval import Hit, Retriever
from news_recommender.storage import Filters, IndexNotReady, Store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
st.set_page_config(page_title="News Explorer", page_icon=":newspaper:", layout="wide")


@st.cache_resource
def services(settings: Settings) -> tuple[Retriever, Chat]:
    retriever = Retriever(settings)
    return retriever, Chat(settings, retriever)


def open_article(article_id: str) -> None:
    st.session_state["selected_article"] = article_id
    st.session_state["page"] = "Search"


def scope_chat(article_id: str | None) -> None:
    st.session_state["chat_article"] = article_id
    st.session_state["messages"] = []
    st.session_state["page"] = "Chat"


def clear_chat() -> None:
    st.session_state["messages"] = []


def show_error(exc: Exception) -> None:
    logger.exception("Application operation failed")
    st.error(str(exc))


ERRORS = (ValueError, OSError, RuntimeError, sqlite3.Error, LookupError, ChromaError)


def result_card(hit: Hit, key: str, label: str | None = None) -> None:
    with st.container(border=True):
        st.subheader(f"{label + ' ' if label else ''}{hit.article.headline}")
        st.caption(
            f"{hit.article.published_at or 'Unknown date'} · "
            f"{hit.article.section or 'Unknown section'} · {hit.article.category}"
        )
        st.text(hit.excerpt[:650] + ("..." if len(hit.excerpt) > 650 else ""))
        st.button(
            "Read full article",
            key=f"read-{key}",
            on_click=open_article,
            args=(hit.article.id,),
        )
        if hit.article.url:
            st.link_button("Original source", hit.article.url)


def metadata_filters(store: Store) -> Filters:
    facets = store.facets()
    category = st.selectbox("Category", ["All", *facets["category"]])
    section = st.selectbox("Section", ["All", *facets["section"]])
    date_from = date_to = None
    if st.checkbox("Filter by publication date"):
        start = st.date_input("From", date(2010, 1, 1))
        end = st.date_input("To", date.today())
        date_from = int(start.strftime("%Y%m%d"))
        date_to = int(end.strftime("%Y%m%d"))
    return Filters(
        category=None if category == "All" else category,
        section=None if section == "All" else section,
        date_from=date_from,
        date_to=date_to,
    )


def search_view(retriever: Retriever, filters: Filters) -> None:
    with st.form("search-form"):
        st.markdown("#### What would you like to read about?")
        question = st.text_input(
            "What would you like to read about?",
            label_visibility="collapsed",
            placeholder="For example: the impact of climate change on coastal cities",
            max_chars=1500,
        )
        submitted = st.form_submit_button("Find articles")
    if submitted:
        st.session_state.pop("selected_article", None)
        st.session_state["results"] = []
        st.session_state["searched"] = False
        st.session_state["result_filters"] = filters
        try:
            with st.spinner("Searching the corpus..."):
                question = question.strip()
                if not question or len(question) > 1500:
                    raise ValueError("Enter a question between 1 and 1500 characters.")
                hits = retriever.search([question], filters, prefer_lexical=True)
            st.session_state["results"] = hits
            st.session_state["searched"] = True
        except ERRORS as exc:
            show_error(exc)
    if st.session_state.get("result_filters") != filters:
        st.session_state["results"] = []
        st.session_state["searched"] = False
    hits = st.session_state.get("results", [])
    if st.session_state.get("searched"):
        if len(hits) < 5:
            st.info(f"Found {len(hits)} article(s). Try another topic or broaden the filters.")
        else:
            st.success("Found 5 articles. Select one to read its complete text.")
        if hits:
            st.caption(
                "Phrase matches first, then all search terms; semantic and partial matches "
                "supplement the results. "
                "Results are ranked candidates, not guaranteed relevant matches. "
            )
    selected = st.session_state.get("selected_article")
    if selected:
        article = retriever.store.article(selected)
        st.divider()
        st.header(article.headline)
        st.caption(f"{article.published_at or 'Unknown date'} · {article.author}")
        if article.url:
            st.link_button("Original source", article.url)
        st.button("Ask about this article", on_click=scope_chat, args=(article.id,))
        st.button(
            "Back to results",
            on_click=lambda: st.session_state.pop("selected_article", None),
        )
        st.text(article.body)
    else:
        for index, hit in enumerate(hits):
            result_card(hit, f"search-{index}-{hit.article.id}")


def chat_view(chat: Chat, filters: Filters) -> None:
    active = st.session_state.get("chat_article")
    if active:
        article = chat.retriever.store.article(active)
        st.info(
            f"Chat scope: {article.headline}. Sidebar filters do not change this article scope."
        )
        st.button("Return to dataset-wide chat", on_click=scope_chat, args=(None,))
        filters = Filters(article_id=active)
    else:
        st.caption("Answers use dataset evidence and the current sidebar filters, not live news.")
    st.button("Clear conversation", on_click=clear_chat)
    messages = st.session_state.setdefault("messages", [])
    for index, message in enumerate(messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("notice"):
                st.caption(message["notice"])
            for number, hit in enumerate(message.get("sources", []), 1):
                with st.expander(f"[{number}] {hit.article.headline}"):
                    st.text(hit.excerpt)
                    st.button(
                        "Open cited article",
                        key=f"source-{index}-{number}",
                        on_click=open_article,
                        args=(hit.article.id,),
                    )
    question = st.chat_input("Ask a news question or a follow-up", max_chars=1500)
    if question:
        history = [{"role": item["role"], "content": item["content"]} for item in messages]
        try:
            with st.spinner("Finding evidence and preparing a cited answer..."):
                reply = chat.answer(question, history, filters)
        except ERRORS as exc:
            show_error(exc)
        else:
            messages.extend(
                [
                    {"role": "user", "content": question},
                    {
                        "role": "assistant",
                        "content": reply.text,
                        "sources": reply.sources,
                        "notice": reply.notice,
                    },
                ]
            )
            st.rerun()


def main() -> None:
    st.title("News Explorer")
    try:
        settings = Settings.from_env()
        store = Store(settings.database)
        store.ensure_ready()
        retriever, chat = services(settings)
        with st.sidebar:
            st.header("Corpus")
            report = store.meta("report")
            if isinstance(report, dict):
                st.write(f"{report['articles']:,} articles · {report['chunks']:,} chunks")
                if report.get("sample_limit"):
                    st.caption(f"Development sample: first {report['sample_limit']} rows")
            filters = metadata_filters(store)
        page = st.radio("View", ["Search", "Chat"], horizontal=True, key="page")
        if page == "Search":
            search_view(retriever, filters)
        else:
            chat_view(chat, filters)
    except IndexNotReady as exc:
        st.info(str(exc))
        st.code("news-ingest --limit 1000\nstreamlit run app.py", language="bash")
    except ERRORS as exc:
        show_error(exc)


main()
