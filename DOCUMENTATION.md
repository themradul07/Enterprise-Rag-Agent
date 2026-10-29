# Enterprise Agentic RAG — Architecture & Codebase Documentation

This document provides a comprehensive, file-by-file breakdown of the **Enterprise Agentic RAG** system. It details every module, the functions and classes it contains, what each does, the libraries leveraged, and all environment variables configured.

---

## 1. Environment Variables Overview (`.env`)

The project uses a [`.env`](file:///d:/PythonProject/EnterpriseRag/.env) file to configure external service credentials and endpoints:

| Variable Name | Purpose & Usage in Codebase |
| :--- | :--- |
| `GROQ_API_KEY` | Primary API key for Groq's high-speed inference engine (used by NeMo Guardrails with `llama-3.1-8b-instant`, LangGraph RAG synthesis, and evaluation judge). |
| `GROQ_FALLBACK_API_KEY` | Secondary/backup Groq API key loaded in configuration for failover support during rate limits. |
| `QDRANT_API_KEY` | API authentication key for the remote Qdrant Cloud vector database cluster. |
| `QDRANT_URL` | Cloud cluster endpoint URL for Qdrant Cloud where dense document embeddings are indexed and searched. |
| `GEMINI_API_KEY` | API key for Google Gemini Generative AI (used for dense document embeddings via `langchain-google-genai`). |

### Additional Environment Variables Referenced Across Code:
- `LOGFIRE_TOKEN`: Token for Pydantic Logfire distributed tracing and observability.
- `JUDGE_GROQ`: Dedicated Groq API key for evaluation judge LLMs (falls back to `GROQ_API_KEY` if not set).
- `PORTKEY_API_KEY`: API key for Portkey AI Gateway routing, caching, and fallback management.
- `BACKEND_URL`: URL of the FastAPI backend used by the Streamlit UI (defaults to `http://localhost:8000`).

---

## 2. File-by-File Breakdown

---

### Root Configuration

#### [`requirements.txt`](file:///d:/PythonProject/EnterpriseRag/requirements.txt)
- **Role:** Declares Python dependencies for the FastAPI backend, LangGraph orchestration, Portkey gateway, Qdrant vector database, FlashRank reranking, NeMo Guardrails, document parsers, Logfire observability, and RAGAS evaluations.
- **Libraries Listed:** `fastapi`, `uvicorn`, `python-dotenv`, `requests`, `streamlit`, `qdrant-client`, `flashrank`, `langchain`, `langgraph`, `langchain-groq`, `nemoguardrails`, `portkey-ai`, `ragas`, `deepeval`, `logfire`, `pypdf`, `python-docx`, `python-pptx`, `beautifulsoup4`, `sentence-transformers`, `langchain-google-genai`.

---

### Core Application (`app/`)

#### [`app/config.py`](file:///d:/PythonProject/EnterpriseRag/app/config.py)
- **Role:** Centralized configuration provider using environment variables.
- **Libraries Used:** `os`, `dotenv` (`python-dotenv`).
- **Classes & Variables:**
  - [`Settings`](file:///d:/PythonProject/EnterpriseRag/app/config.py#L5-L12): Loads and encapsulates `QDRANT_API_KEY`, `QDRANT_URL`, `GEMINI_API_KEY`, `QDRANT_COLLECTION` ("enterprise_rag"), `GROQ_API_KEY`, `GROQ_MODEL` ("llama-3.3-70b-versatile"), and `GROQ_FALLBACK_API_KEY`.
  - `settings`: Singleton instance of `Settings`.

#### [`app/__init__.py`](file:///d:/PythonProject/EnterpriseRag/app/__init__.py)
- **Role:** Package initialization exposing default `Settings` instance.
- **Libraries Used:** `os`, `dotenv`.
- **Classes & Variables:**
  - `Settings`: Defines configuration attributes for the package namespace.

#### [`app/main.py`](file:///d:/PythonProject/EnterpriseRag/app/main.py)
- **Role:** FastAPI application entrypoint exposing REST API endpoints, initializing guardrails, and executing the LangGraph agent workflow.
- **Libraries Used:** `fastapi`, `pydantic`, `logfire`, `dotenv`.
- **Functions & Endpoints:**
  - [`startup_event()`](file:///d:/PythonProject/EnterpriseRag/app/main.py#L24-L26): FastAPI startup hook that calls `initialize_rails()` to prepare NeMo guardrails.
  - [`home()`](file:///d:/PythonProject/EnterpriseRag/app/main.py#L32-L34) (`GET /`): Health check endpoint verifying that the API is online.
  - [`get_graph_image()`](file:///d:/PythonProject/EnterpriseRag/app/main.py#L37-L46) (`GET /graph`): Draws and returns a Mermaid PNG diagram representing the LangGraph workflow structure.
  - [`query(request: QueryRequest)`](file:///d:/PythonProject/EnterpriseRag/app/main.py#L49-L101) (`POST /query`): Executes the two-gate pipeline:
    1. **Gate 1:** Checks the question through NeMo Guardrails (`guard()`). If triggered (jailbreak/off-topic), immediately returns the canned response.
    2. **Gate 2:** Invokes the LangGraph RAG workflow (`rag_agent.invoke()`) with thread memory and returns the answer, reasoning steps, and retrieved sources.

---

### Agentic Orchestration (`app/agents/`)

#### [`app/agents/state.py`](file:///d:/PythonProject/EnterpriseRag/app/agents/state.py)
- **Role:** Defines the shared state schema for LangGraph.
- **Libraries Used:** `typing` (`TypedDict`, `List`, `Annotated`), `operator`.
- **Classes:**
  - [`AgentState`](file:///d:/PythonProject/EnterpriseRag/app/agents/state.py#L5-L12): Typed dictionary with fields: `message` (accumulated conversation messages via `operator.add`), `current_query`, `documents`, `plan`, `status`, and `final_answer`.

#### [`app/agents/graph.py`](file:///d:/PythonProject/EnterpriseRag/app/agents/graph.py)
- **Role:** Constructs and compiles the LangGraph cyclic state machine workflow with checkpoint memory.
- **Libraries Used:** `langgraph` (`StateGraph`, `END`, `MemorySaver`).
- **Functions & Objects:**
  - [`route_planner(state)`](file:///d:/PythonProject/EnterpriseRag/app/agents/graph.py#L17-L23): Inspects `state["current_query"]`; routes directly to `responder` if marked "CONVERSATIONAL", otherwise routes to `retriever`.
  - `rag_agent`: Compiled LangGraph application configured with `checkpointer=MemorySaver()` for conversational session memory.

#### [`app/agents/nodes/planner.py`](file:///d:/PythonProject/EnterpriseRag/app/agents/nodes/planner.py)
- **Role:** Determines query intent (conversational greeting/memory lookup vs. technical RAG search) and refines search terms.
- **Libraries Used:** `logfire`, internal gateway (`get_langchain_llm`).
- **Functions:**
  - [`planner_node(state)`](file:///d:/PythonProject/EnterpriseRag/app/agents/nodes/planner.py#L6-L51): Formats conversation history, prompts the LLM to classify if fresh retrieval is needed or if it can be answered conversationally, and updates `current_query`, `status`, and `plan`.

#### [`app/agents/nodes/retriever.py`](file:///d:/PythonProject/EnterpriseRag/app/agents/nodes/retriever.py)
- **Role:** Executes hybrid retrieval combining vector similarity search with cross-encoder reranking.
- **Libraries Used:** `logfire`, internal services (`search_enterprise_knowledge`, `rerank_documents`).
- **Functions:**
  - [`retrieve_node(state)`](file:///d:/PythonProject/EnterpriseRag/app/agents/nodes/retriever.py#L6-L30): Retrieves top 15 candidate chunks from Qdrant vector database, runs FlashRank semantic cross-encoder reranking to keep the top 5 most relevant chunks, and updates state documents.

#### [`app/agents/nodes/responder.py`](file:///d:/PythonProject/EnterpriseRag/app/agents/nodes/responder.py)
- **Role:** Synthesizes the final answer using retrieved technical context, conversation history, and Portkey gateway client with cache-hit detection.
- **Libraries Used:** `logfire`, internal gateway (`portkey_client`, `extract_cache_status`).
- **Functions:**
  - [`generate_node(state)`](file:///d:/PythonProject/EnterpriseRag/app/agents/nodes/responder.py#L5-L86): Prompts the LLM via Portkey's client, inspects HTTP response headers for cache hits (`x-portkey-cache-status`), updates the agent plan with cache status, and formats the final answer.

---

### LLM Gateway (`app/gateway/`)

#### [`app/gateway/__init__.py`](file:///d:/PythonProject/EnterpriseRag/app/gateway/__init__.py)
- **Role:** Exposes gateway clients and utility methods.
- **Exports:** `portkey_client`, `get_langchain_llm`, `extract_cache_status`.

#### [`app/gateway/client.py`](file:///d:/PythonProject/EnterpriseRag/app/gateway/client.py)
- **Role:** Configures Portkey AI Gateway for unified LLM routing, automatic fallbacks (`llama-3.3-70b-versatile` -> `llama-3.1-8b-instant`), retries (on 429/503), and caching.
- **Libraries Used:** `portkey-ai` (`Portkey`, `createHeaders`, `PORTKEY_GATEWAY_URL`), `langchain-openai` (`ChatOpenAI`), `logfire`.
- **Functions & Objects:**
  - `GATEWAY_CONFIG`: Dict defining fallback strategy, simple caching mode, retry policies, and virtual model targets.
  - `portkey_client`: Native `Portkey` client instance.
  - [`get_langchain_llm(feature)`](file:///d:/PythonProject/EnterpriseRag/app/gateway/client.py#L30-L55): Returns a Portkey-proxied `ChatOpenAI` instance compatible with LangChain components.
  - [`extract_cache_status(response)`](file:///d:/PythonProject/EnterpriseRag/app/gateway/client.py#L57-L68): Defensively parses `x-portkey-cache-status` from HTTP response headers to identify "HIT" or "MISS".

---

### Guardrails (`app/guardrails/`)

#### [`app/guardrails/__init__.py`](file:///d:/PythonProject/EnterpriseRag/app/guardrails/__init__.py)
- **Role:** Exposes guardrails initialization and checking functions.
- **Exports:** `initialize_rails`, `guard`.

#### [`app/guardrails/colang_rules.py`](file:///d:/PythonProject/EnterpriseRag/app/guardrails/colang_rules.py)
- **Role:** Colang and YAML rule definitions for NVIDIA NeMo Guardrails.
- **Libraries Used:** None (pure Python string configurations).
- **Variables:**
  - `COLANG_CONTENT`: Flow specifications for off-topic filtering, jailbreak attempts, greetings, capabilities, and farewells.
  - `YAML_CONTENT`: General instructions and bot persona guidelines.
  - `RAIL_INDICATORS`: List of canonical substrings indicating a rail intervention has occurred.

#### [`app/guardrails/rails.py`](file:///d:/PythonProject/EnterpriseRag/app/guardrails/rails.py)
- **Role:** NeMo Guardrails execution engine using `llama-3.1-8b-instant`.
- **Libraries Used:** `nemoguardrails` (`RailsConfig`, `LLMRails`), `langchain-groq` (`ChatGroq`), `logfire`.
- **Functions:**
  - [`initialize_rails()`](file:///d:/PythonProject/EnterpriseRag/app/guardrails/rails.py#L11-L32): Instantiates the singleton `LLMRails` object using Groq's fast 8B model.
  - [`guard(message)`](file:///d:/PythonProject/EnterpriseRag/app/guardrails/rails.py#L34-L61): Tests incoming queries against Colang rails. Returns `(True, response)` if blocked or dialog intercepted; returns `(False, None)` if clean.

---

### Document Ingestion & Parsers (`app/ingestion/`)

#### [`app/ingestion/__init__.py`](file:///d:/PythonProject/EnterpriseRag/app/ingestion/__init__.py)
- **Role:** Package marker for the ingestion subsystem.

#### [`app/ingestion/processor.py`](file:///d:/PythonProject/EnterpriseRag/app/ingestion/processor.py)
- **Role:** Main ingestion pipeline that scans directories, detects formats, parses documents, splits chunks, creates Qdrant collections, embeds texts, and indexes vectors.
- **Libraries Used:** `qdrant-client`, `uuid`, `json`, `os`, `sys`, `logfire`.
- **Functions:**
  - [`save_processed_locally(data, source_type, filename)`](file:///d:/PythonProject/EnterpriseRag/app/ingestion/processor.py#L29-L36): Writes parsed and chunked text metadata to `processed_data/<source_type>/<filename>.json`.
  - [`process_file(file_path, filename, source_type)`](file:///d:/PythonProject/EnterpriseRag/app/ingestion/processor.py#L39-L100): Parses a single file, chunks it, saves JSON locally, generates embeddings, and upserts points to Qdrant.
  - [`process_directory(dir_path, source_type)`](file:///d:/PythonProject/EnterpriseRag/app/ingestion/processor.py#L102-L109): Iterates over all files in a folder and calls `process_file()`.
  - [`run_universal_ingestion(base_dir, explicit_source_type, wipe)`](file:///d:/PythonProject/EnterpriseRag/app/ingestion/processor.py#L111-L166): Coordinates full directory ingestion, supports collection recreation/wipe via `--wipe`, and resolves embedding vector dimensions.

#### [`app/ingestion/chunking/splitter.py`](file:///d:/PythonProject/EnterpriseRag/app/ingestion/chunking/splitter.py)
- **Role:** Semantic paragraph-aware chunker.
- **Libraries Used:** `typing`, `logfire`.
- **Functions:**
  - [`chunk_text(text, chunk_size=1500)`](file:///d:/PythonProject/EnterpriseRag/app/ingestion/chunking/splitter.py#L4-L31): Splits text on double-newlines (`\n\n`) and groups paragraphs up to the maximum character threshold (`chunk_size`).

#### [`app/ingestion/loaders/pdf.py`](file:///d:/PythonProject/EnterpriseRag/app/ingestion/loaders/pdf.py)
- **Role:** Extracts text from PDF files locally.
- **Libraries Used:** `pypdf` (`PdfReader`), `pdfplumber` (as graceful fallback for image-heavy/blank pages), `logfire`.
- **Functions:**
  - [`parse_pdf(file_path)`](file:///d:/PythonProject/EnterpriseRag/app/ingestion/loaders/pdf.py#L5-L52): Iterates through PDF pages extracting text via `pypdf`; triggers `pdfplumber` fallback for blank pages.

#### [`app/ingestion/loaders/html.py`](file:///d:/PythonProject/EnterpriseRag/app/ingestion/loaders/html.py)
- **Role:** Cleans and parses HTML documents.
- **Libraries Used:** `bs4` (`BeautifulSoup`), `logfire`.
- **Functions:**
  - [`parse_html(file_path)`](file:///d:/PythonProject/EnterpriseRag/app/ingestion/loaders/html.py#L4-L32): Strips non-content tags (`<script>`, `<style>`, `<meta>`, `<noscript>`) and normalizes whitespace into clean plain text.

#### [`app/ingestion/loaders/office.py`](file:///d:/PythonProject/EnterpriseRag/app/ingestion/loaders/office.py)
- **Role:** Extracts text from Microsoft Office files (`.docx`, `.pptx`).
- **Libraries Used:** `unstructured` (`unstructured.partition.auto.partition`), `logfire`.
- **Functions:**
  - [`parse_office(file_path)`](file:///d:/PythonProject/EnterpriseRag/app/ingestion/loaders/office.py#L4-L24): Automatically partitions Word and PowerPoint files and combines parsed elements into plain text.

#### [`app/ingestion/loaders/text.py`](file:///d:/PythonProject/EnterpriseRag/app/ingestion/loaders/text.py)
- **Role:** Reads plain text files (`.txt`, `.md`).
- **Libraries Used:** Built-in Python I/O, `logfire`.
- **Functions:**
  - [`parse_text(file_path)`](file:///d:/PythonProject/EnterpriseRag/app/ingestion/loaders/text.py#L3-L14): Opens and reads UTF-8 plain text files with error tolerance.

---

### Retrieval & Ranking Services (`app/services/retrieval/`)

#### [`app/services/retrieval/embeddings.py`](file:///d:/PythonProject/EnterpriseRag/app/services/retrieval/embeddings.py)
- **Role:** Dense embedding generation service powered by local Sentence-Transformers (`all-mpnet-base-v2`).
- **Libraries Used:** `sentence-transformers` (`SentenceTransformer`), `logfire`.
- **Functions:**
  - `_get_model()`: Lazily initializes the local `all-mpnet-base-v2` Sentence-Transformers model singleton (768 dimensions).
  - [`get_embedding_dim()`](file:///d:/PythonProject/EnterpriseRag/app/services/retrieval/embeddings.py#L18-L20): Returns the vector dimension (`768`).
  - [`embed_query(query)`](file:///d:/PythonProject/EnterpriseRag/app/services/retrieval/embeddings.py#L25-L30): Generates a single 768-dim vector for a user query.
  - [`embed_texts(texts)`](file:///d:/PythonProject/EnterpriseRag/app/services/retrieval/embeddings.py#L32-L41): Batch encodes multiple text chunks in batches of 50.

#### [`app/services/retrieval/qdrant_service.py`](file:///d:/PythonProject/EnterpriseRag/app/services/retrieval/qdrant_service.py)
- **Role:** Vector database interaction for semantic nearest-neighbor search.
- **Libraries Used:** `qdrant-client` (`QdrantClient`, `models`), `logfire`.
- **Functions:**
  - [`search_enterprise_knowledge(query, limit=8)`](file:///d:/PythonProject/EnterpriseRag/app/services/retrieval/qdrant_service.py#L13-L39): Converts query text into a vector using `embed_query()`, queries Qdrant via `query_points()`, and formats matching payloads and similarity scores.

#### [`app/services/retrieval/ranking_service.py`](file:///d:/PythonProject/EnterpriseRag/app/services/retrieval/ranking_service.py)
- **Role:** Cross-encoder reranker refining top Qdrant vector candidates.
- **Libraries Used:** `flashrank` (`Ranker`, `RerankRequest`), `time`, `logfire`.
- **Functions:**
  - `_get_ranker()`: Lazily instantiates FlashRank's local quantized ONNX cross-encoder model (`ms-marco-MiniLM-L-6-v2`).
  - [`rerank_documents(query, documents, top_n=5)`](file:///d:/PythonProject/EnterpriseRag/app/services/retrieval/ranking_service.py#L23-L63): Scores and reorders documents against the query semantically, returning the top N most relevant passages.

---

### Evaluation Suite (`evals/`)

#### [`evals/__init__.py`](file:///d:/PythonProject/EnterpriseRag/evals/__init__.py)
- **Role:** Package exports for evaluation pipeline and metric modules.

#### [`evals/pipeline.py`](file:///d:/PythonProject/EnterpriseRag/evals/pipeline.py)
- **Role:** Phase 1 of evaluation: queries the live FastAPI `/query` endpoint with golden dataset questions and collects live responses.
- **Libraries Used:** `requests`, `time`, `copy`, `json`, `logfire`.
- **Functions:**
  - [`detect_tool(thought_process)`](file:///d:/PythonProject/EnterpriseRag/evals/pipeline.py#L24-L39): Parses the backend's reasoning steps to infer which tool was invoked (`guardrails`, `retrieve_documents`, `direct_answer`).
  - [`run_pipeline(golden_dataset, progress_callback)`](file:///d:/PythonProject/EnterpriseRag/evals/pipeline.py#L41-L106): Sends each golden sample to `/query`, captures truncated responses and retrieved sources, and enriches the dataset.
  - [`save_results(dataset, path)`](file:///d:/PythonProject/EnterpriseRag/evals/pipeline.py#L108-L111): Persists enriched evaluation results to a JSON file.
  - [`load_golden_dataset()`](file:///d:/PythonProject/EnterpriseRag/evals/pipeline.py#L113-L117): Loads `evals/golden_dataset.json`.

#### [`evals/guardrails_eval.py`](file:///d:/PythonProject/EnterpriseRag/evals/guardrails_eval.py)
- **Role:** Binary safety evaluation testing adversarial and legitimate prompts against the live guardrail system.
- **Libraries Used:** `requests`, `time`, `copy`, `logfire`.
- **Functions:**
  - `_is_blocked(response_json)`: Checks if the response reasoning steps confirm that guardrails fired.
  - [`run_guardrails_eval(guardrails_samples, progress_callback)`](file:///d:/PythonProject/EnterpriseRag/evals/guardrails_eval.py#L21-L79): Evaluates inputs against `/query` and classifies each as True Positive (TP), True Negative (TN), False Positive (FP), or False Negative (FN).
  - [`compute_guardrails_metrics(results)`](file:///d:/PythonProject/EnterpriseRag/evals/guardrails_eval.py#L81-L99): Calculates precision, recall, and accuracy across test cases.

#### [`evals/metrics.py`](file:///d:/PythonProject/EnterpriseRag/evals/metrics.py)
- **Role:** Phase 2 of evaluation: evaluates system quality using the RAGAS framework with a dedicated Groq judge model.
- **Libraries Used:** `ragas` (`Faithfulness`, `AnswerRelevancy`, `ContextPrecision`, `ContextRecall`, `AnswerCorrectness`, `llm_factory`, `HuggingFaceEmbeddings`), `openai` (`AsyncOpenAI`), `pandas`, `asyncio`, `logfire`.
- **Functions:**
  - `_build_judge()`: Configures an asynchronous Groq client for `llama-3.1-8b-instant` and HuggingFace mini embeddings.
  - `_cooldown(seconds, label)`: Asynchronous sleep helper enforcing Groq TPM compliance.
  - `_prep_samples(golden_dataset)`: Filters and truncates contexts to stay within token-per-minute limits.
  - `_score_df(metric_key, samples, scores)`: Formats evaluation results into pandas DataFrames.
  - `_batched_score(metric, inputs, samples)`: Batches asynchronous metric scoring calls with cooldown intervals.
  - [`run_all_metrics(golden_dataset, status_cb)`](file:///d:/PythonProject/EnterpriseRag/evals/metrics.py#L98-L229): Runs all 6 evaluation experiments: Faithfulness, Answer Relevancy, Context Precision, Context Recall, Answer Correctness, and Tool Correctness (Jaccard similarity).

#### [`evals/data_parser.py`](file:///d:/PythonProject/EnterpriseRag/evals/data_parser.py)
- **Role:** Direct parser for ground truth and noisy data files using lightweight native libraries.
- **Libraries Used:** `python-docx` (`docx`), `python-pptx` (`pptx`), `os`, `sys`, and internal text/HTML loaders.
- **Functions:**
  - `_parse_docx(file_path)`: Extracts text from Word documents via `python-docx`.
  - `_parse_pptx(file_path)`: Extracts slide text from PowerPoint files via `python-pptx`.
  - [`parse_file(file_path)`](file:///d:/PythonProject/EnterpriseRag/evals/data_parser.py#L39-L53): Dispatches parsing to appropriate loaders based on extension.
  - [`load_all_chunks()`](file:///d:/PythonProject/EnterpriseRag/evals/data_parser.py#L55-L84): Parses all files in `true_data` and `noisy_data` folders, tagging each chunk with `is_noise`.

#### [`evals/app.py`](file:///d:/PythonProject/EnterpriseRag/evals/app.py)
- **Role:** Multi-step Streamlit dashboard for managing and visualizing the evaluation lifecycle.
- **Libraries Used:** `streamlit`, `pandas`, `nest_asyncio`, `logfire`.
- **Functions:**
  - `_badge(score)`, `_grade(score)`, `_color_score(val)`: UI styling helpers for metric score color-coding (Good / Fair / Poor).
  - `_render_metric_table(df, metric_col, title)`: Renders styled Streamlit dataframes with averages and badges.
  - `_run_async(coro)`: Executes asynchronous coroutines within Streamlit's synchronous event loop.

---

### User Interface (`ui/`)

#### [`ui/app.py`](file:///d:/PythonProject/EnterpriseRag/ui/app.py)
- **Role:** Streamlit chat interface for local execution.
- **Libraries Used:** `streamlit`, `requests`, `uuid`, `time`, `logfire`, `python-dotenv`.
- **Key Features & Logic:**
  - Manages session state (`session_id` and conversation history).
  - Provides sidebar with memory management and Logfire status indicator.
  - Sends user queries via HTTP POST to `/query`.
  - Streams synthesized responses character-by-character.
  - Surfaces backend reasoning steps (`thought_process`) and expandable source chunks.

#### [`ui/st_cloud_ui.py`](file:///d:/PythonProject/EnterpriseRag/ui/st_cloud_ui.py)
- **Role:** Streamlit chat interface optimized for cloud deployment (e.g., Streamlit Community Cloud).
- **Libraries Used:** `streamlit`, `requests`, `uuid`, `time`, `logfire`, `os`.
- **Key Features & Logic:**
  - Configures Logfire using `st.secrets` fallback to `os.getenv`.
  - Enables `logfire.instrument_requests()` for cross-service distributed tracing propagation to FastAPI.
  - Provides chat messaging, reasoning step visibility, and source context preview.
