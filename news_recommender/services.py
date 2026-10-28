import logging
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Literal

from .chat import Chat, ChatError
from .config import Settings
from .retrieval import Retriever
from .storage import Filters, Store

logger = logging.getLogger(__name__)
ServiceName = Literal["retrieval", "chat"]


def report_startup_failure(future: Future[Retriever] | Future[Chat]) -> None:
    error = future.exception()
    if error is not None:
        logger.error(
            "Background service initialization failed",
            exc_info=(type(error), error, error.__traceback__),
        )


class Services:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = Store(settings.database)
        self.store.ensure_ready()
        self._lock = Lock()
        self._stages: dict[ServiceName, str] = {
            "retrieval": "Opening keyword and vector indexes...",
            "chat": "Waiting for retrieval models...",
        }
        executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="news-startup")
        self.retrieval = executor.submit(self._load_retrieval)
        self.chat = executor.submit(self._load_chat)
        self.retrieval.add_done_callback(report_startup_failure)
        self.chat.add_done_callback(report_startup_failure)
        executor.shutdown(wait=False)

    def stage(self, service: ServiceName) -> str:
        with self._lock:
            return self._stages[service]

    def _set_stage(self, service: ServiceName, stage: str) -> None:
        with self._lock:
            self._stages[service] = stage
        logger.info("%s startup: %s", service, stage)

    def _load_retrieval(self) -> Retriever:
        try:
            retriever = Retriever(self.settings, store=self.store)
            self._set_stage("retrieval", "Loading embedding model and reranker...")
            retriever.nlp.warmup()
            self._set_stage("retrieval", "Warming keyword and vector search...")
            retriever.search(
                ["news"], Filters(), limit=self.settings.search_limit, prefer_lexical=True
            )
            return retriever
        finally:
            self.store.close()

    def _load_chat(self) -> Chat:
        retriever = self.retrieval.result()
        self._set_stage(
            "chat",
            f"Loading Ollama model ({self.settings.context_tokens:,}-token context)...",
        )
        chat = Chat(self.settings, retriever)
        try:
            chat.warmup()
        except ChatError as exc:
            chat.warmup_warning = str(exc)
            logger.warning("%s", exc)
        return chat
