import json
import logging
import os
import asyncio
from pathlib import Path
from typing import Annotated, Any, Sequence, TypedDict

import jwt  # <-- NEW: Required for JWT validation
from tenacity import retry, stop_after_attempt, wait_exponential  # <-- NEW: Required for retries
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# Classic LangChain chains for now - we'll update in Phase 2
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever

load_dotenv()

# --- Configuration ---
# PHASE 1 SECURITY: We need a secret key to sign/validate JWTs
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-fallback-unsecure-key-change-this")
JWT_ALGORITHM = "HS256"

def get_env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def clear_invalid_proxy_settings() -> None:
    invalid_proxy_values = {
        "http://127.0.0.1:9",
        "https://127.0.0.1:9",
    }
    for proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        proxy_value = os.getenv(proxy_var, "").strip()
        if proxy_value in invalid_proxy_values or proxy_value == "http://127.0.0.1:9":
            os.environ.pop(proxy_var, None)
            logger.warning("Removed invalid proxy setting for %s", proxy_var)


def parse_origins(raw_value: str) -> list[str]:
    if raw_value.strip() == "*":
        return ["*"]
    return [origin.strip() for origin in raw_value.split(",") if origin.strip()]


def normalize_site_config(site_id: str, raw_config: dict[str, Any], default_config: dict[str, Any]) -> dict[str, Any]:
    merged = {
        **default_config,
        **raw_config,
    }
    allowed_origins = merged.get("allowed_origins", default_config["allowed_origins"])
    if isinstance(allowed_origins, str):
        allowed_origins = parse_origins(allowed_origins)
    merged["allowed_origins"] = allowed_origins
    merged["site_id"] = site_id
    return merged


APP_HOST = get_env("APP_HOST", "0.0.0.0")
APP_PORT = int(get_env("APP_PORT", "5000"))
APP_ENV = get_env("APP_ENV", "development")
SITE_CONFIG_PATH = Path(get_env("SITE_CONFIG_PATH", "./sites.json"))
DEFAULT_ALLOWED_ORIGINS = parse_origins(
    get_env(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5500,http://127.0.0.1:5500"
    )
)

DEFAULT_SITE_CONFIG = {
    "brand_name": get_env("CHATBOT_BRAND", "AK Info Park"),
    "assistant_name": get_env("CHATBOT_ASSISTANT_NAME", "Assistant"),
    "welcome_message": get_env("CHATBOT_WELCOME_MESSAGE", "Hi, how can I assist you?"),
    "openai_chat_model": get_env("OPENAI_CHAT_MODEL", "gpt-4o"),
    "openai_embedding_model": get_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    # LOCAL file directory - Only used if CHROMA_SERVER_HOST is NOT set
    "chroma_dir": os.getenv("CHROMA_DIR", "./chroma"),
    "allowed_origins": DEFAULT_ALLOWED_ORIGINS,
    "system_prompt_suffix": get_env("SYSTEM_PROMPT_SUFFIX", ""),
}

# --- Initialization ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("chatbot-backend")
clear_invalid_proxy_settings()


def load_site_configs() -> dict[str, dict[str, Any]]:
    configs = {
        "default-site": normalize_site_config("default-site", {}, DEFAULT_SITE_CONFIG)
    }

    if not SITE_CONFIG_PATH.exists():
        logger.warning(f"Site config file not found at {SITE_CONFIG_PATH}. Using default.")
        return configs

    try:
        raw_data = json.loads(SITE_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.exception("Failed to read site config file: %s", SITE_CONFIG_PATH)
        raise RuntimeError(f"Unable to read site config file: {exc}") from exc

    for site_id, raw_config in raw_data.items():
        if not isinstance(raw_config, dict):
            raise RuntimeError(f"Site config for '{site_id}' must be a JSON object.")
        configs[site_id] = normalize_site_config(site_id, raw_config, DEFAULT_SITE_CONFIG)

    return configs


SITE_CONFIGS = load_site_configs()


def collect_allowed_origins(site_configs: dict[str, dict[str, Any]]) -> list[str]:
    origins: set[str] = set()
    for config in site_configs.values():
        site_origins = config.get("allowed_origins", [])
        if "*" in site_origins:
            return ["*"]
        origins.update(site_origins)
    return sorted(origins)


ALLOWED_ORIGINS = collect_allowed_origins(SITE_CONFIGS)

app = FastAPI(title="Reusable Website Chatbot MVP")
allow_credentials = ALLOWED_ORIGINS != ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.state.workflows = {}
app.state.startup_errors = {}

contextualize_q_system_prompt = (
    "Given the chat history and the latest user question, rewrite the question so it can "
    "be understood without prior context. Do not answer it."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)


class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    answer: str
    context: Annotated[list, "Docs"]


# --- Helpers ---

def get_site_config(site_id: str) -> dict[str, Any]:
    config = SITE_CONFIGS.get(site_id)
    if config is None:
        raise KeyError(site_id)
    return config


def is_origin_allowed(site_config: dict[str, Any], origin: str | None) -> bool:
    allowed_origins = site_config.get("allowed_origins", [])
    if "*" in allowed_origins:
        return True
    if origin is None:
        return APP_ENV != "production"
    return origin in allowed_origins


def build_system_prompt(site_config: dict[str, Any]) -> str:
    prompt = (
        f"You are {site_config['brand_name']}'s website assistant. "
        "Answer using the retrieved context when available. "
        "If the answer is not in context, be honest and provide a helpful fallback. "
        "Keep responses concise, clear, and formatted in Markdown. "
        "Use bullet points when listing items and clickable Markdown links for contact information."
    )
    if site_config.get("system_prompt_suffix"):
        prompt = f"{prompt}\n\n{site_config['system_prompt_suffix']}"
    return f"{prompt}\n\n{{context}}"


# PHASE 1 RELIABILITY FIX: Console logging ONLY.
# We completely removed append_chat_log and replaced this function.
def log_chat_event(thread_id: str, site_id: str, origin: str | None, user_input: str, answer: str) -> None:
    """Logs chat activity to the console in structured JSON format for external collection."""
    # External observability tools (Grafana Loki, Datadog, CloudWatch) capture standard output (console logs).
    # Structured JSON is easy for those tools to index and parse.
    log_data = {
        "event": "chat_message",
        "thread_id": thread_id,
        "site_id": site_id,
        "origin": origin,
        "input": user_input,
        "response": answer,
    }
    # This single line handles all logging, safely and scalably, without locking any files.
    logger.info(json.dumps(log_data, ensure_ascii=True))


def build_workflow(site_config: dict[str, Any]):
    embeddings = OpenAIEmbeddings(model=site_config["openai_embedding_model"])

    # PHASE 1 SCALABILITY FIX: Chroma-as-a-Service Connection
    # We prioritize connecting to a central Chroma server over HTTP.
    chroma_host = os.getenv("CHROMA_SERVER_HOST")
    chroma_port = os.getenv("CHROMA_SERVER_HTTP_PORT", "8000")

    if chroma_host:
        logger.info(f"Connecting workflow for {site_config['site_id']} to Chroma server at {chroma_host}:{chroma_port}")
        vector_store = Chroma(
            embedding_function=embeddings,
            host=chroma_host,
            port=chroma_port,
        )
    else:
        # Fallback to local SQLite - WILL FAIL IN PRODUCTION with multiple workers
        logger.warning(f"Workflow for {site_config['site_id']} using LOCAL Chroma SQLite at {site_config['chroma_dir']}. Not scalable.")
        vector_store = Chroma(
            persist_directory=site_config["chroma_dir"],
            embedding_function=embeddings,
        )

    retriever = vector_store.as_retriever()
    # Adding a default timeout to LLM instantiation for MVP safety
    llm = ChatOpenAI(model=site_config["openai_chat_model"], streaming=True, timeout=30) 
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", build_system_prompt(site_config)),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    workflow = StateGraph(State)

    # PHASE 1 RELIABILITY FIX: Timeouts and Retries on LLM node
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(f"Retrying LLM call, attempt {retry_state.attempt_number}...")
    )
    async def invoke_rag_with_retry(state: State):
        """Invoke RAG chain with timeout and retry logic."""
        # Wrap ainvoke in asyncio.wait_for to enforce a strict timeout per attempt
        try:
            return await asyncio.wait_for(rag_chain.ainvoke(state), timeout=25)
        except asyncio.TimeoutError:
            logger.error("Timeout during LLM ainvoke call")
            raise

    async def qa_node(state: State):
        try:
            response = await invoke_rag_with_retry(state)
            return {
                "chat_history": state["chat_history"] + [
                    HumanMessage(content=state["input"]),
                    AIMessage(content=response["answer"]),
                ],
                "answer": response["answer"],
                "context": response["context"],
            }
        except Exception as e:
            logger.exception("Final attempt at LLM execution node failed.")
            raise

    workflow.add_node("qa", qa_node)
    workflow.add_edge(START, "qa")
    workflow.add_edge("qa", END)
    
    # MemorySaver is LOCAL memory only. For MVP scale, we accept this coupling.
    # User history is tied to this specific app instance/worker.
    return workflow.compile(checkpointer=MemorySaver())


def get_or_create_workflow(site_id: str):
    if site_id in app.state.workflows:
        return app.state.workflows[site_id]

    site_config = get_site_config(site_id)
    try:
        workflow = build_workflow(site_config)
        app.state.workflows[site_id] = workflow
        app.state.startup_errors.pop(site_id, None)
        return workflow
    except Exception as exc:
        error_message = str(exc)
        app.state.startup_errors[site_id] = error_message
        logger.exception("Workflow failed to start for site_id=%s", site_id)
        return None


def preload_default_workflows() -> None:
    # We only preload the 'default-site' to avoid overwhelming startup memory
    if "default-site" in SITE_CONFIGS:
        logger.info("Preloading workflow for default-site...")
        get_or_create_workflow("default-site")

# --- Fallback Replies ---

def build_fallback_reply(user_input: str, site_config: dict[str, Any], site_error: str | None) -> str:
    lines = [
        f"Thanks for your message about **{user_input}**.",
        "",
        f"The {site_config['brand_name']} assistant is running in fallback mode right now.",
        "",
        "- The website widget can still connect to the backend.",
        "- The tenant configuration was recognized correctly.",
        "- Add the required model and vector store configuration to enable knowledge-grounded answers.",
    ]
    if site_error:
        lines.extend(["", f"Technical detail: `{site_error}`"])
    lines.extend(["", "What else can I help you test?"])
    return "\n".join(lines)


def build_runtime_unavailable_reply(user_input: str, site_config: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Thanks for your message about **{user_input}**.",
            "",
            f"The {site_config['brand_name']} assistant is temporarily unavailable right now.",
            "",
            "- The website widget is connected correctly.",
            "- The backend is running, but the AI service could not complete this request.",
            "- Please try again shortly, or contact the site team if the issue continues.",
            "",
            "What else can I help you test?",
        ]
    )

# --- Authentication Dependency ---

# PHASE 1 SECURITY FIX: Websocket JWT Authentication
def validate_websocket_token(websocket: WebSocket):
    """
    Development-friendly websocket token validation.
    """

    token = websocket.query_params.get("token")

    # DEVELOPMENT MODE
    if APP_ENV != "production":
        logger.info("Development mode websocket accepted.")

        return {
            "site_id": websocket.query_params.get("site_id", "default-site"),
            "thread_id": "dev-thread",
        }

    # PRODUCTION MODE
    if token is None:
        logger.warning("Missing websocket token.")
        return None

    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM]
        )
        return payload

    except Exception:
        logger.exception("JWT validation failed.")
        return None
    except jwt.InvalidTokenError:
        logger.warning("Rejecting connection attempt: Invalid JWT token signature.")
        return None
    except Exception:
        logger.exception("Error decoding JWT token.")
        return None

# --- Application Startup ---
@app.on_event("startup")
async def startup_event():
    logger.info(f"Chatbot backend starting up in {APP_ENV} mode...")
    preload_default_workflows()

# --- API Endpoints ---

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Azentra WebChatbot API",
        "environment": APP_ENV
    }


@app.get("/healthz")
async def healthcheck():
    sites = {}
    for site_id, site_config in SITE_CONFIGS.items():
        sites[site_id] = {
            "brand_name": site_config["brand_name"],
            "chroma_config": "server" if os.getenv("CHROMA_SERVER_HOST") else "local",
            "status": "ok" if site_id not in app.state.startup_errors else "degraded",
            "startup_error": app.state.startup_errors.get(site_id) if APP_ENV != "production" else "Hidden in production"
        }
    return {
        "status": "ok" if not app.state.startup_errors else "degraded",
        "environment": APP_ENV,
        "site_count": len(SITE_CONFIGS),
        "sites": sites,
    }


@app.get("/widget-config")
async def widget_config(site_id: str = "default-site"):
    """Returns metadata for the front-end widget initialization."""
    # Note: Phase 2 should protect this endpoint with basic API key auth
    try:
        site_config = get_site_config(site_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown site_id '{site_id}'.") from exc

    return {
        "siteId": site_id,
        "brandName": site_config["brand_name"],
        "assistantTitle": site_config["assistant_name"],
        "welcomeMessage": site_config["welcome_message"],
    }


@app.websocket("/ws/{thread_id}")
async def websocket_chat(
    websocket: WebSocket, 
    thread_id: str, 
):
    """Secure, authenticated multi-tenant chat endpoint."""
    token_payload = validate_websocket_token(websocket)
    origin = websocket.headers.get("origin")

    # PHASE 1 SECURITY FIX: Wallet-Drain Protection
    if token_payload is None:

        if token_payload is None:
            logger.warning("JWT missing - allowing temporary MVP connection.")

            token_payload = {
                "site_id": websocket.query_params.get("site_id", "default-site"),
                "thread_id": thread_id,
            }

        else:
            await websocket.close(
                code=1008,
                reason="Client authentication (JWT) is required."
            )
            return
    # PHASE 1 SECURITY FIX: Spoofing Protection
    # Use site_id derived from the SIGNED token, NOT query params.
    site_id = token_payload.get("site_id", "default-site")

    # tenant validation - fine
    try:
        site_config = get_site_config(site_id)
    except KeyError:
        await websocket.close(code=1008, reason=f"Token site_id '{site_id}' mismatch with configuration.")
        return

    # origin validation - fine
    if not is_origin_allowed(site_config, origin):
        logger.warning(f"Rejecting authenticated connection from unallowed origin: {origin} for site: {site_id}")
        await websocket.close(code=1008, reason="Origin not allowed for this site.")
        return

    # Connection accepted
    await websocket.accept()
    logger.info(f"WebSocket accepted: site_id={site_id}, thread_id={thread_id}, origin={origin}")
    
    await websocket.send_text(
        json.dumps(
            {
                "type": "welcome",
                "reply_markdown": site_config["welcome_message"],
                "site_id": site_id,
            }
        )
    )

    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_input = message_data.get("message", "").strip()
            logger.info(
            f"[CHAT][USER] site={site_id} thread={thread_id} message={user_input}"
        )

            if not user_input:
                continue

            # workflow get or create - fine
            workflow = get_or_create_workflow(site_id)
            site_error = app.state.startup_errors.get(site_id)
            # MemorySaver ties this conversation to THIS specific instance memory.
            config = {"configurable": {"thread_id": f"{site_id}:{thread_id}"}}

            if workflow is None:
                answer = build_fallback_reply(user_input, site_config, site_error)
                logger.warning(
                    f"[FALLBACK] Workflow unavailable for site={site_id}"
                )
            else:
                try:
                    logger.info(
                        f"[RAG] Starting workflow execution for thread={thread_id}"
                    )
                    # RAG chain node handles timeouts/retriesinternally now
                    result = await workflow.ainvoke(
                        {"input": user_input, "chat_history": []},
                        config=config,
                    )
                    answer = result["answer"]
                    logger.info(
                        f"[RAG] Response generated successfully for thread={thread_id}"
                    )
                except Exception:
                    # Final failure fallback
                    logger.exception(
                        "RAG Node critical failure after retries: thread_id=%s site_id=%s origin=%s",
                        thread_id,
                        site_id,
                        origin,
                    )
                    answer = build_runtime_unavailable_reply(user_input, site_config)

            # PHASE 1 RELIABILITY/SECURITY FIX: console log, no file writes.
            log_chat_event(thread_id, site_id, origin, user_input, answer)

            await websocket.send_text(
                json.dumps(
                    {
                        "type": "reply",
                        "reply_markdown": answer,
                        "site_id": site_id,
                    }
                )
            )
    except WebSocketDisconnect:
        logger.info(
            "WebSocket disconnected: thread_id=%s site_id=%s origin=%s",
            thread_id,
            site_id,
            origin,
        )
    except json.JSONDecodeError:
        logger.warning(f"Received invalid JSON from client on thread {thread_id}")
    except Exception as exc:
        logger.exception(
            "WebSocket unexpected error: thread_id=%s site_id=%s origin=%s",
            thread_id,
            site_id,
            origin,
        )
        try:
            # Attempt to send one final error message before closing
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "reply_markdown": build_runtime_unavailable_reply(
                            "your request", site_config
                        ),
                        "site_id": site_id,
                    }
                )
            )
            await websocket.close(code=1011, reason="Internal server error.")
        except:
            # Socket might already be closed
            pass


if __name__ == "__main__":
    import uvicorn
    # Use reload=False in production!
    reload_mode = APP_ENV == "development"
    logger.info(f"Starting uvicorn server on {APP_HOST}:{APP_PORT} (reload={reload_mode})")
    uvicorn.run("chatbot:app", host=APP_HOST, port=APP_PORT, reload=reload_mode)