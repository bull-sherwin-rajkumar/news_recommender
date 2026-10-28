# News Explorer

A small Python app for **hybrid full-text/semantic news search, complete-article reading,
and dataset-grounded chat** over [CNN-english-news](https://huggingface.co/datasets/shivam9980/CNN-english-news).
It runs inference locally and needs no database server or paid API.

## Setup

Python 3.12+, SQLite with FTS5, and [Ollama](https://ollama.com/) are required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
ollama pull gemma4:12b  # skip if already installed
```

Start `ollama serve` if Ollama is not already running. The initial dataset and model
downloads need internet access. Thereafter search and inference use local data/models.
Embeddings use CUDA and half precision by default. Install a CUDA-enabled PyTorch
build compatible with your GPU/driver; CUDA unavailability is an explicit error.
Set `NEWS_DEVICE=cpu` for CPU embeddings or `auto` for PyTorch device selection.
The reranker defaults to CPU ONNX (`NEWS_RERANKER_BACKEND=onnx`), keeping GPU
memory available for Ollama. Initial startup downloads the ONNX model into the
Hugging Face cache; subsequent offline startup loads the cached local snapshot.
ONNX model overrides must supply `onnx/model.onnx` and tokenizer/config files,
or point to an equivalent local model directory. Use `NEWS_RERANKER_BACKEND=torch`
to run other cross-encoders with PyTorch on `NEWS_DEVICE` (half precision on CUDA).
The 12B LLM is substantial: allow space
for its model, the PyTorch/model caches, and the corpus indexes.

After the initial model downloads, you can prevent Hugging Face startup metadata
checks as well:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 streamlit run app.py
```

## Index and run

```bash
# Development: first 1,000 source rows (invalid/duplicate rows may reduce article count).
news-ingest --limit 1000
streamlit run app.py
```

Ingestion downloads the pinned dataset once into the Hugging Face datasets cache,
then iterates over local Arrow data instead of a network stream. Even `--limit`
downloads the source dataset on the first run; subsequent runs reuse it.

For the complete corpus (37,949 source rows at the pinned revision):

```bash
# Prefer a separate directory if you want to keep the sample.
NEWS_DATA_DIR=./data-full news-ingest
NEWS_DATA_DIR=./data-full streamlit run app.py
```

Alternatively, explicitly replace the sample using `news-ingest --rebuild` with no
`--limit`. A rebuild replaces only this app's SQLite records and Chroma collection.
Re-running ingestion with identical settings resumes interrupted batches and does
not add duplicate articles/vectors. Stop the frontend while ingesting or rebuilding.
Do not run multiple frontend processes against this embedded index.

The ingestion log reports rejected rows, duplicate/existing rows, malformed/missing
dates, ready article/chunk counts, vector counts, and publication-date coverage.
It checks the expected source-row count for the pinned dataset revision before
declaring ingestion complete.
Search is blocked until ingestion completes successfully. Changing the dataset
limit, embedding model, or chunk settings requires a rebuild or another data directory.
Changing the Ollama model does not require reindexing.

## Using it

- Enter a topic or question and optionally filter category, section, and dates.
- Search uses your original input without an LLM rewrite. Full-query phrase
  matches come first, followed by matches containing all meaningful search terms.
  Partial keyword and semantic matches supplement those results, not displace them.
- Browse **up to 300 ranked candidates**, ten per page, using the compact row of page
  numbers, Previous, and Next controls below the results. `NEWS_SEARCH_LIMIT` can
  change the cap (5–500). The displayed count is the retrieved candidate count,
  not the total number of corpus matches. New searches and filter changes reset
  pagination. Scores are not probabilities
  or a guarantee of relevance; semantic search can return loosely related articles.
- Click **Read full article** to fetch the original complete stored body.
- Use **Chat** for cited dataset-grounded answers and conversational follow-ups.
  Answers stream as drafts; citations are validated at completion. Rejected drafts
  are cleared before a retry, and failed/interrupted drafts never enter history.
- Select **Ask about this article** to scope chat to that article. This resets the
  conversation; return to dataset-wide chat to remove the scope.
- Chat history lasts only for the current Streamlit session. There are no accounts,
  favorites, personalization, or political-stance ranking.

This is a historical corpus, not live news. The assistant reports insufficient
evidence instead of deliberately answering from general model knowledge.
Search and article reading do not require a running Ollama server. Conversational
query interpretation and answer generation in Chat require Ollama.

## How retrieval works

1. Ingestion retains complete original articles in SQLite, alongside cleaned indexing
   text. Stable parent IDs and exact body/URL deduplication prevent duplicate records.
2. Paragraph/sentence-aware chunks contain at most 320 embedding-tokenizer tokens,
   with about 64 tokens of overlap. Token offsets refer to the **normalized indexing
   text**, not the original reader body. Long sentences are split on token boundaries.
3. A bounded headline prefix accompanies chunks. Embedding inputs are checked against
   actual tokenizer limits. The reranker tokenizes each pair once in `predict` and
   truncates to its configured 512-token maximum.
4. SQLite FTS5 uses BM25, stronger headline/keyword weights, Unicode normalization,
   and English stemming. User text is converted to escaped terms, not raw FTS syntax.
   Search retrieves the full query as a phrase, then all meaningful terms together,
   then any-term matches. Phrase matching keeps intervening stop words; all-term
   matching ignores common stop words and allows terms across the indexed headline,
   keywords, and one passage. It does not require all terms to be adjacent.
5. Chroma stores normalized, 384-dimensional `BAAI/bge-small-en-v1.5` embeddings.
   Queries use the model's retrieval instruction.
6. Lexical and vector candidates are collapsed by parent article before reciprocal
   rank fusion, so long articles are not rewarded for having more chunks.
   Repeated CNN photo-gallery/UI passages are ignored as retrieval evidence to avoid
   recommending unrelated stories that embed the same gallery. The reader still
   retains the original full article.
   Search protects phrase and all-term matches before candidate truncation, so
   vector candidates cannot crowd them out of the reranker shortlist.
   Both lexical and semantic retrieval are bounded: by default, Search fetches at
   most 3,000 chunks per lexical tier and 500 nearest-neighbor chunks, then keeps
   up to 300 article candidates. Chat retains smaller retrieval budgets.
   Candidate metadata is fetched in one `WHERE id IN (...)` query without original
   article bodies. Cleaned full indexing text is loaded only for explicit year or
   quoted-phrase constraints. Reader/scoped-chat requests fetch full bodies by ID.
7. `cross-encoder/ms-marco-MiniLM-L6-v2` reranks at most **20 articles with one
   evidence passage each**, using CPU ONNX by default. Its raw
   scores order candidates without a hard score cutoff, in both Search and Chat.
   A small redundancy penalty encourages variety.
   In Search, scores and diversity only reorder articles within the same lexical
   tier: phrase, all terms, or supplemental. Chat keeps standard hybrid fusion
   and reranking without this lexical-first policy.
   Search reranks the leading shortlist, then retains remaining candidates up to
   the result cap in fusion order within each lexical tier. These
   remaining candidates have fusion scores, not cross-encoder scores. This keeps
   model inference bounded independently of the result cap.
   Explicit years and quoted phrases must occur in the source; publication-date filtering
   remains a separate control. This avoids substituting an older election for a requested year.
8. In Chat, first-turn questions go directly to retrieval without an LLM rewrite.
   Follow-ups still use Ollama `gemma4:12b` to resolve conversational references.
   Standalone literal search queries must fit the existing 96-token query budget.
   Ollama then streams plain-text answers from supplied, source-labeled evidence;
   JSON schema output is used only for follow-up query planning. Search passes
   the original query directly to retrieval. The LLM does not write replacement
   news or decide vector ranks.
   Unsupported topics may still retrieve loosely related vector candidates; Chat
   must refuse an answer when the supplied evidence is insufficient. Citation
   validation and model-refusal handling remain in place.

Streamlit shows the page and article reader without waiting for model warmup.
Cached background workers open the retrieval indexes, warm the embedding model
and reranker, and run a representative Search request through keyword retrieval,
vector retrieval, and reranking before enabling Search. This moves first-query
index initialization into startup; warmup results are discarded, not displayed
or saved in search history. Workers then ask Ollama to load the LLM with
`keep_alive="30m"`. Temporary
loading messages appear only in the selected Search or Chat view and disappear
when ready; there is no Startup sidebar section. Search becomes
available as soon as retrieval is ready, independently of Ollama; Chat input is
disabled until its warmup finishes. Startup runs once per configuration/server
process, not on page refresh. Every subsequent LLM request refreshes the retention
period. An Ollama warmup failure is logged and shown as a warning without blocking
Search or article reading; Chat can still retry with a normal answer request.
After changing model/service initialization code, fully restart Streamlit to clear
previously imported modules and cached initialization failures; refreshing the
browser alone does not clear them.
The default context remains **64,000 tokens**; an existing `NEWS_CONTEXT_TOKENS`
environment value still overrides it. This is the configured maximum, not the
actual prompt length. Check `ollama ps` and available VRAM: a 64k KV cache plus
model weights and the embedding model may require CPU offload on a smaller GPU.

SQLite connections are reused per thread. The ready-state cache checks SQLite's
`data_version` to notice commits from other connections, so an active rebuild
still blocks reads. Chroma's vector count is cached for search and invalidated by
vector writes/reset. Stop the frontend before changing the embedded index, as above.

The reader uses an article ID to load the full article directly, **not** concatenated
chunks. For article chat, the complete article is supplied when it fits the conservative
context budget; otherwise the app discloses that only retrieved excerpts were used.
Valid source labels are enforced, but that alone does not prove every answer is correct.

## Modules

- [config](news_recommender/config.py): environment configuration and index identity.
- [storage](news_recommender/storage.py): canonical articles/chunks, FTS5, and Chroma.
- [nlp](news_recommender/nlp.py): normalization, token-aware chunking, embeddings, reranking.
- [ingestion](news_recommender/ingestion.py): bounded dataset ingestion and consistency checks.
- [retrieval](news_recommender/retrieval.py): hybrid parent-level retrieval and relevance selection.
- [chat](news_recommender/chat.py): query interpretation, contextual answers, source validation.
- [services](news_recommender/services.py): background startup and model warmup progress.
- [frontend](app.py): Streamlit search, reader, and conversational UI.

See [.env.example](.env.example) for paths, model overrides, timeout, context budget, and
device selection. `.env` is read automatically; explicit environment variables win.
If workspace policy prevents creating or reading `.env`, use explicit environment
variables and set `PYTHON_DOTENV_DISABLED=1`.
`NEWS_MIN_SCORE` and the evaluation CLI's `--threshold` option have been removed.
Remove `NEWS_MIN_SCORE` from existing environment files; it no longer controls
retrieval. Existing indexes remain compatible and do not need rebuilding.
Labels in [tests/relevance_labels.json](tests/relevance_labels.json) retain their
historical `calibration` and `evaluation` split names for compatibility; they no
longer imply threshold tuning. Use a separate full-corpus data directory as documented.

## FTS5 exploration notebook

[experimentation.ipynb](experimentation.ipynb) exposes raw SQLite FTS5 `MATCH`
syntax against the full local corpus. Install interactive notebook support in
the selected kernel environment with `pip install -e '.[notebook]'`, then run the
setup/helper cells. `matches = match_search('"labor shortage"')` returns **every**
matching chunk in BM25 order, with clickable page numbers below the output.
Use `page_size=20` to change the page size without limiting the total. Notebook
results are chunks, so an article can appear more than once; the app instead
deduplicates by article. Rerun the helper and example cells after updating.

## Validation

```bash
pytest
ruff check .
python -m compileall -q news_recommender app.py
# See the evaluation command's help for a labeled-query file format.
news-evaluate --help
PYTHON_DOTENV_DISABLED=1 NEWS_DATA_DIR=./data \
  news-evaluate --labels tests/relevance_labels.json --split calibration
PYTHON_DOTENV_DISABLED=1 NEWS_DATA_DIR=./data-full \
  news-evaluate --labels tests/relevance_labels.json --split evaluation
```

Normal tests use deterministic fixtures and doubles; they require neither downloads nor
a running LLM. Model-backed evaluation should be run separately against a ready index.
Inspect precision, ranking quality, unsupported queries, and answer citation support;
do not infer quality from successful API calls alone.
The evaluation CLI measures retrieval, not generated answers. Its unsupported-query
abstention metrics measure whether retrieval returns no candidates, not whether Chat
refuses an unsupported answer. Ungated retrieval may return candidates for such queries.
By default it evaluates standard hybrid retrieval without LLM rewriting, rather than
the Search UI's lexical-first policy (`Retriever.search(..., prefer_lexical=True)`).

### Verified corpus

At the pinned revision, complete ingestion examined **37,949 rows**, retained
**37,912 unique usable articles**, rejected 26 invalid rows, and excluded 11 exact
body/URL duplicates. It produced **207,654 chunks**, with identical SQLite, FTS5, and
persisted Chroma counts. The complete index occupied approximately **1.9 GiB** in
this environment. Stored publication dates span August 2011 through March 2022.
Rerunning a partially completed ingestion also counts previously indexed rows in
`duplicates_or_existing`; that counter is not solely the number of source duplicates.

The test suite covers crash recovery, source completeness, token coverage/budgets,
bounded retrieval/reranking, batch metadata loading, per-thread connection reuse,
ready-state invalidation, streaming completion/failure handling, citations,
full-article reading, and the frontend search/reader/chat interaction.
Actual local-model smoke checks also exercised
grounded answers, follow-ups, and the search-to-complete-article flow.

### Historical small retrieval benchmark (before cutoff removal)

The figures below were recorded under the former threshold-gated hybrid policy.
They have not been rerun for ungated retrieval and do not validate the current
Search or Chat retrieval behavior.

The included labels contain six development-sample calibration cases and 14
full-corpus evaluation cases: ten topical queries and four unsupported queries.
Grades are 0 = irrelevant, 1 = partially relevant, 2 = directly relevant. Judgments
were made from pooled result titles, descriptions, and passage evidence.

| Method | Precision@5 | nDCG@5 | Unsupported queries correctly rejected |
|---|---:|---:|---:|
| BM25 only | 0.933 | 0.779 | 3/4 |
| Hybrid + cross-encoder | 1.000 | 0.909 | 4/4 |

Precision@5 covers the nine topical cases with at least five labeled relevant
articles; nDCG covers all ten topical cases. The sparse SpaceX case returned only
two articles rather than padding to five, and also exposed missed relevant evidence.
This was a **small, manually pooled development benchmark**, not an exhaustive or
blind independent evaluation. The evaluation queries were separate from threshold
calibration but also helped diagnose repeated gallery noise. Reassess precision,
recall, and answer support on additional queries before making general quality claims.

## Data and attribution

Keep downloaded data, generated indexes, and model caches out of source control.
The app retains article metadata and source URLs for attribution.
The Hugging Face repository declares Apache-2.0, but this does **not** independently
establish rights to redistribute the underlying CNN articles. Review source rights
before publishing or redistributing a corpus or deployed reader.


## Start the app
PYTHON_DOTENV_DISABLED=1 NEWS_DATA_DIR=./data-full \
  .venv/bin/streamlit run app.py