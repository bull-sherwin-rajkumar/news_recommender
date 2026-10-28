import logging
import sqlite3
from datetime import date

import streamlit as st
from chromadb.errors import ChromaError

from news_recommender.chat import Chat
from news_recommender.config import Settings
from news_recommender.retrieval import Hit, Retriever
from news_recommender.services import ServiceName, Services
from news_recommender.storage import Filters, IndexNotReady, Store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
st.set_page_config(page_title="News Explorer", page_icon=":newspaper:", layout="wide")


@st.cache_resource(show_spinner=False)
def services(settings: Settings) -> Services:
    return Services(settings)


def service_status(startup: Services, service: ServiceName) -> None:
    future = startup.retrieval if service == "retrieval" else startup.chat
    label = "Search" if service == "retrieval" else "Chat"
    if not future.done():
        st.info(f"{label}: {startup.stage(service)}")
    elif future.exception() is not None:
        st.error(f"{label} initialization failed: {future.exception()}")
    elif service == "chat" and startup.chat.result().warmup_warning:
        st.warning(startup.chat.result().warmup_warning)


@st.fragment(run_every="1s")
def poll_service_status(startup: Services, service: ServiceName) -> None:
    service_status(startup, service)
    future = startup.retrieval if service == "retrieval" else startup.chat
    if future.done():
        st.rerun()


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
SEARCH_PAGE_SIZE = 10


def set_search_page(page: int) -> None:
    st.session_state["search_page"] = page


def search_pagination(page: int, page_count: int) -> None:
    numbers = sorted({1, page_count, *range(max(1, page - 2), min(page_count, page + 2) + 1)})
    slots: list[int | None] = []
    previous = 0
    for number in numbers:
        if previous and number - previous > 1:
            slots.append(None)
        slots.append(number)
        previous = number
    with st.container(horizontal=True, gap="small", vertical_alignment="center"):
        st.button(
            "Previous",
            disabled=page == 1,
            on_click=set_search_page,
            args=(page - 1,),
        )
        for number in slots:
            if number is None:
                st.markdown("...", width="content")
            else:
                st.button(
                    str(number),
                    key=f"search-page-{number}",
                    type="primary" if number == page else "secondary",
                    disabled=number == page,
                    on_click=set_search_page,
                    args=(number,),
                )
        st.button(
            "Next",
            disabled=page == page_count,
            on_click=set_search_page,
            args=(page + 1,),
        )


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


def search_view(retriever: Retriever | None, filters: Filters, store: Store) -> None:
    with st.form("search-form"):
        st.markdown("#### What would you like to read about?")
        question = st.text_input(
            "What would you like to read about?",
            label_visibility="collapsed",
            placeholder="For example: the impact of climate change on coastal cities",
            max_chars=1500,
        )
        submitted = st.form_submit_button("Find articles", disabled=retriever is None)
    if submitted:
        st.session_state.pop("selected_article", None)
        st.session_state["results"] = []
        st.session_state["searched"] = False
        st.session_state["search_page"] = 1
        st.session_state["result_filters"] = filters
        try:
            if retriever is None:
                raise RuntimeError("Search models are still loading. Check the startup status.")
            with st.spinner("Searching the corpus..."):
                question = question.strip()
                if not question or len(question) > 1500:
                    raise ValueError("Enter a question between 1 and 1500 characters.")
                hits = retriever.search(
                    [question], filters, limit=retriever.settings.search_limit, prefer_lexical=True
                )
            st.session_state["results"] = hits
            st.session_state["searched"] = True
        except ERRORS as exc:
            show_error(exc)
    if st.session_state.get("result_filters") != filters:
        st.session_state["results"] = []
        st.session_state["searched"] = False
        st.session_state["search_page"] = 1
    hits = st.session_state.get("results", [])
    if st.session_state.get("searched"):
        if not hits:
            st.info("Found 0 articles. Try another topic or broaden the filters.")
        else:
            st.success(f"Found {len(hits):,} articles. Select one to read its complete text.")
    selected = st.session_state.get("selected_article")
    if selected:
        article = store.article(selected)
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
        page_count = max(1, (len(hits) + SEARCH_PAGE_SIZE - 1) // SEARCH_PAGE_SIZE)
        page = min(max(1, st.session_state.get("search_page", 1)), page_count)
        start = (page - 1) * SEARCH_PAGE_SIZE
        for index, hit in enumerate(hits[start : start + SEARCH_PAGE_SIZE], start=start):
            result_card(hit, f"search-{index}-{hit.article.id}")
        if hits:
            st.caption(
                f"Showing {start + 1}–{min(start + SEARCH_PAGE_SIZE, len(hits))} "
                f"of {len(hits):,} articles · Page {page} of {page_count}"
            )
            search_pagination(page, page_count)


def chat_view(chat: Chat | None, filters: Filters, store: Store) -> None:
    active = st.session_state.get("chat_article")
    if active:
        article = store.article(active)
        st.info(
            f"Chat scope: {article.headline}. Sidebar filters do not change this article scope."
        )
        st.button("Return to dataset-wide chat", on_click=scope_chat, args=(None,))
        filters = Filters(article_id=active)
    else:
        st.caption("Answers are based on the current dataset version and the sidebar filters.")
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
    question = st.chat_input(
        "Ask a news question or a follow-up", max_chars=1500, disabled=chat is None
    )
    if question:
        history = [{"role": item["role"], "content": item["content"]} for item in messages]
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            draft_notice = st.empty()
            draft_notice.caption(
                "Generating a draft; source citations are checked when it finishes."
            )
            draft = st.empty()

        def show_draft(text: str) -> None:
            draft.markdown(text)

        try:
            if chat is None:
                raise RuntimeError("Chat is still warming up. Check the startup status.")
            with st.spinner("Finding evidence and preparing a cited answer..."):
                reply = chat.answer(question, history, filters, on_text=show_draft)
        except ERRORS as exc:
            draft.empty()
            draft_notice.empty()
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
        startup = services(settings)
        store = startup.store
        store.ensure_ready()
        with st.sidebar:
            st.header("Corpus")
            report = store.meta("report")
            if isinstance(report, dict):
                st.write(f"{report['articles']:,} articles · {report['chunks']:,} chunks")
                if report.get("sample_limit"):
                    st.caption(f"Development sample: first {report['sample_limit']} rows")
            filters = metadata_filters(store)
        retrieval_done = startup.retrieval.done()
        chat_done = startup.chat.done()
        retriever = (
            startup.retrieval.result()
            if retrieval_done and startup.retrieval.exception() is None
            else None
        )
        chat = (
            startup.chat.result()
            if chat_done and startup.chat.exception() is None
            else None
        )
        page = st.radio("View", ["Search", "Chat"], horizontal=True, key="page")
        selected_service: ServiceName = "retrieval" if page == "Search" else "chat"
        selected_done = retrieval_done if page == "Search" else chat_done
        if selected_done:
            service_status(startup, selected_service)
        else:
            poll_service_status(startup, selected_service)
        if page == "Search":
            search_view(retriever, filters, store)
        else:
            chat_view(chat, filters, store)
    except IndexNotReady as exc:
        st.info(str(exc))
        st.code("news-ingest --limit 1000\nstreamlit run app.py", language="bash")
    except ERRORS as exc:
        show_error(exc)


main()
