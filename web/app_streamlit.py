import base64
import gc
import hashlib
import html
import math
import os
import re
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable, Optional, TypedDict, cast

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

st.set_page_config(page_title="DocuSun UI", layout="wide")


CUSTOM_CSS = """
<style>
:root {
    --bg: #070d16;
    --surface: #111a26;
    --surface-2: #0d141f;
    --line: rgba(245, 97, 92, 0.45);
    --text: #f3f7ff;
    --muted: #b8c3d8;
}

.stApp {
    background: radial-gradient(1200px 500px at 0% 0%, #14233a 0%, var(--bg) 45%);
    color: var(--text);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a2130 0%, #141b28 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

div[data-testid="stChatInput"] textarea {
    border: 1px solid var(--line) !important;
    background: rgba(255, 255, 255, 0.03) !important;
}

div[data-testid="stStatusWidget"] {
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
}

.st-key-clear_chat_btn {
    margin-top: 2rem;
}

h1, h2, h3, p, span, label {
    color: var(--text);
}

small {
    color: var(--muted);
}
</style>
"""


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "file_cache" not in st.session_state:
    st.session_state.file_cache = {}
if "active_file_key" not in st.session_state:
    st.session_state.active_file_key = None
if "messages" not in st.session_state:
    st.session_state.messages = []


class CachedDocument(TypedDict):
    file_name: str
    file_key: str
    persist_dir: str
    workdir: str
    file_bytes: bytes
    index_steps: list[str]


class RagSettings(TypedDict):
    chunk_size: int
    tokenization_model: str
    embedding_model: str
    embedding_device: str
    llm_model: str
    llm_display_name: str
    llm_provider: str
    llm_api_key: str
    top_k: int
    nvidia_api_key: str
    google_api_key: str


class LlmOption(TypedDict):
    model: str
    provider: str


TOKENIZATION_MODEL_MAP: dict[str, str] = {
    "gpt2": "gpt2",
    "gemma-2b": "google/gemma-2b",
}

EMBEDDING_MODEL_MAP: dict[str, str] = {
    "nvidia-v3-335m": "nvidia/nv-embedqa-e5-v5",
    "llama-1b-v2": "nvidia/llama-nemotron-embed-1b-v2",
}

LLM_MODEL_OPTIONS: dict[str, LlmOption] = {
    "qwen3-80b-instruct": {
        "model": "qwen/qwen3-next-80b-a3b-instruct",
        "provider": "nvidia",
    },
    "mistral-medium-3.5": {
        "model": "mistralai/mistral-medium-3.5-128b",
        "provider": "nvidia",
    },
    "gemini-2.5-flash": {
        "model": "gemini-2.5-flash",
        "provider": "api",
    },
    "llama-3.1-70b": {
        "model": "meta/llama-3.1-70b-instruct",
        "provider": "nvidia",
    },
    
}

NVIDIA_API_KEY_ENV = "NVIDIA_API_KEY"
GOOGLE_API_KEY_ENV = "GOOGLE_API_KEY"
NVIDIA_PROVIDER = "nvidia"
GOOGLE_PROVIDER = "api"
DEFAULT_TOP_K = 3
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_SIZE_STEP = 50
DEFAULT_CHUNK_OVERLAP_RATIO = 0.15
INDEXING_STEP_SEQUENCE = [
    "Discovering files",
    "Loading documents",
    "Splitting into chunks",
    "Persisting embeddings",
]


def _get_file_cache() -> dict[str, CachedDocument]:
    cache = st.session_state.get("file_cache")
    if not isinstance(cache, dict):
        cache = {}
        st.session_state.file_cache = cache
    return cast(dict[str, CachedDocument], cache)



def _to_env_bool(value: bool) -> str:
    return "true" if value else "false"


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_feature_enabled(env_names: tuple[str, ...], default: bool = False) -> bool:
    for name in env_names:
        if name in os.environ:
            return _parse_bool(os.environ.get(name), default=default)
    return default


def _persist_feature_flags(updates: dict[str, bool], env_path: Path) -> None:
    if not updates:
        return

    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    remaining = set(updates.keys())
    for i, line in enumerate(lines):
        for name in tuple(remaining):
            if re.match(rf"^\s*{re.escape(name)}\s*=", line):
                hash_pos = line.find("#")
                comment = ""
                if hash_pos != -1:
                    comment_text = line[hash_pos + 1 :].strip()
                    if comment_text:
                        comment = f"  # {comment_text}"
                lines[i] = f"{name}={_to_env_bool(updates[name])}{comment}"
                remaining.remove(name)
                break

    if remaining:
        if lines and lines[-1].strip():
            lines.append("")
        for name in sorted(remaining):
            lines.append(f"{name}={_to_env_bool(updates[name])}")

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name, enabled in updates.items():
        os.environ[name] = _to_env_bool(enabled)


def _sync_retrieval_feature_flags(flags: dict[str, tuple[str, ...]], values: dict[str, bool]) -> bool:
    pending_updates: dict[str, bool] = {}
    for feature_name, env_names in flags.items():
        enabled = values[feature_name]
        for env_name in env_names:
            current_value = _parse_bool(os.environ.get(env_name), default=False)
            if current_value != enabled:
                pending_updates[env_name] = enabled

    if not pending_updates:
        return False

    _persist_feature_flags(pending_updates, PROJECT_ROOT / ".env")
    st.session_state.file_cache = {}
    st.session_state.active_file_key = None
    return True


def reset_chat() -> None:
    st.session_state.messages = []
    gc.collect()


def display_pdf(pdf_bytes: bytes) -> None:
    st.markdown("### PDF Preview")
    try:
        # Prefer Streamlit's native PDF widget. It handles larger files
        # better than embedding base64 data URLs in an iframe.
        st.pdf(pdf_bytes, height=900)  # type: ignore[attr-defined]
        return
    except Exception:
        pass

    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_display = (
        f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
        'width="100%" height="900" type="application/pdf" '
        'style="border:none; border-radius:8px;">'
        "</iframe>"
    )
    st.markdown(pdf_display, unsafe_allow_html=True)


def _build_file_key(filename: str, content: bytes, settings: RagSettings) -> str:
    digest = hashlib.sha1(content).hexdigest()  # nosec B324
    settings_key = "|".join(
        [
            f"chunk_size={settings['chunk_size']}",
            f"tokenization_model={settings['tokenization_model']}",
            f"embedding_model={settings['embedding_model']}",
            f"embedding_device={settings['embedding_device']}",
            f"top_k={settings['top_k']}",
        ]
    )
    settings_digest = hashlib.sha1(settings_key.encode("utf-8")).hexdigest()[:12]  # nosec B324
    return f"{filename}:{digest}:settings{settings_digest}"


def _stream_words(text: str):
    words = text.split(" ")
    for i, word in enumerate(words):
        suffix = " " if i < len(words) - 1 else ""
        yield f"{word}{suffix}"


def _format_elapsed_label(elapsed_seconds: float) -> str:
    total_seconds = max(0, int(round(elapsed_seconds)))
    minutes, seconds = divmod(total_seconds, 60)
    return f"[{minutes:02d}:{seconds:02d}]"


def _render_assistant_content(
    content: str,
    elapsed_seconds: float | None = None,
    model_hint: str | None = None,
    target=None,
    with_cursor: bool = False,
) -> None:
    safe_content = html.escape(content).replace("\n", "<br>")
    if with_cursor:
        safe_content += "▌"

    elapsed_html = ""
    if isinstance(elapsed_seconds, (int, float)):
        label = _format_elapsed_label(float(elapsed_seconds))
        elapsed_html = (
            "<span style='position:absolute; top:0; right:0; "
            "font-family:monospace; color:#b8c3d8;'>"
            f"{label}</span>"
        )

    model_hint_html = ""
    content_bottom_padding = "0.0rem"
    if model_hint:
        safe_model_hint = html.escape(model_hint)
        content_bottom_padding = "1.15rem"
        model_hint_html = (
            "<span style='position:absolute; bottom:0; right:0; "
            "font-size:0.82rem; color:#9fb0cc; font-family:monospace; text-align:right;'>"
            f"{safe_model_hint}</span>"
        )

    renderer = target.markdown if target is not None else st.markdown
    renderer(
        (
            "<div style='position:relative; padding-right:5.4rem; "
            f"padding-bottom:{content_bottom_padding}; min-height:2.3rem;'>"
            f"{elapsed_html}"
            f"{model_hint_html}"
            f"<div style='white-space:pre-wrap;'>{safe_content}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _build_index_step_logger(
    status_container,
) -> tuple[Callable[[str, str, object | None], None], Callable[[], None]]:
    completed_steps: set[str] = set()
    dynamic_steps: list[str] = []
    current_step: str | None = None
    steps_placeholder = status_container.empty()

    def _ordered_steps() -> list[str]:
        merged = list(INDEXING_STEP_SEQUENCE)
        for step in dynamic_steps:
            if step not in merged:
                merged.append(step)
        return merged

    def _render() -> None:
        lines: list[str] = []
        for step in _ordered_steps():
            if step in completed_steps:
                lines.append(f":green[✓ {step}]")
            else:
                lines.append(f":yellow[• {step}]")
        steps_placeholder.markdown("\n\n".join(lines))

    def _log(kind: str, message: str, value=None) -> None:
        nonlocal current_step
        if kind != "step":
            return
        if message not in INDEXING_STEP_SEQUENCE and message not in dynamic_steps:
            dynamic_steps.append(message)
        if current_step and current_step != message:
            completed_steps.add(current_step)
        current_step = message
        _render()

    def _finalize() -> None:
        nonlocal current_step
        if current_step:
            completed_steps.add(current_step)
            current_step = None
        _render()

    _render()
    return _log, _finalize


def _prepare_document(
    uploaded_file,
    settings: RagSettings,
    log_fn: Optional[Callable[[str, str, object | None], None]] = None,
) -> CachedDocument:
    file_bytes = uploaded_file.getvalue()
    file_key = _build_file_key(uploaded_file.name, file_bytes, settings)
    file_cache = _get_file_cache()

    if file_key in file_cache:
        cached_doc = file_cache[file_key]
        if log_fn:
            cached_steps = cached_doc.get("index_steps", [])
            if cached_steps:
                for step in cached_steps:
                    log_fn("step", step, None)
            else:
                log_fn("step", "Using cached index", None)
        st.session_state.active_file_key = file_key
        return cached_doc

    workdir = tempfile.mkdtemp(prefix=f"docusun_ui_{st.session_state.session_id}_")
    data_dir = os.path.join(workdir, "data")
    persist_dir = os.path.join(workdir, "chroma")
    os.makedirs(data_dir, exist_ok=True)

    pdf_path = os.path.join(data_dir, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(file_bytes)

    index_steps: list[str] = []

    def _capture_and_log(kind: str, message: str, value=None) -> None:
        if kind == "step":
            index_steps.append(message)
        if log_fn:
            log_fn(kind, message, value)

    try:
        from engine.qa.pipeline import index_documents

        index_documents(
            data_path=data_dir,
            chunk_size=settings["chunk_size"],
            top_k=settings["top_k"],
            persist_directory=persist_dir,
            tokenizer_model=settings["tokenization_model"],
            embedding_model=settings["embedding_model"],
            embedding_device=settings["embedding_device"],
            embedding_provider=NVIDIA_PROVIDER,
            embedding_api_key=settings["nvidia_api_key"],
            log_fn=_capture_and_log,
        )
    except Exception:
        raise

    payload: CachedDocument = {
        "file_name": uploaded_file.name,
        "file_key": file_key,
        "persist_dir": persist_dir,
        "workdir": workdir,
        "file_bytes": file_bytes,
        "index_steps": index_steps,
    }
    file_cache[file_key] = payload
    st.session_state.active_file_key = file_key
    return payload


def _answer_question(question: str, persist_dir: str, settings: RagSettings) -> str:
    from engine.qa.pipeline import query_documents

    return query_documents(
        question=question,
        top_k=settings["top_k"],
        persist_directory=persist_dir,
        embedding_model=settings["embedding_model"],
        embedding_device=settings["embedding_device"],
        embedding_provider=NVIDIA_PROVIDER,
        embedding_api_key=settings["nvidia_api_key"],
        llm_model=settings["llm_model"],
        llm_provider=settings["llm_provider"],
        llm_api_key=settings["llm_api_key"],
    )


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


with st.sidebar:
    st.markdown(
        (
            "<div style='display:flex; align-items:flex-end; gap:0.35rem; "
            "margin:0.2rem 0 1.6rem 0;'>"
            "<span style='font-size:3.1rem; font-weight:500; line-height:1.05; "
            "letter-spacing:0.01em;'>DocuSun</span>"
            "<span style='font-size:1rem; font-style:italic; font-weight:500; "
            "line-height:1.2; color:#d8dfef;'>Beta</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    st.markdown("### Upload Document")
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type=["pdf"])
    st.markdown("### Settings")

    with st.expander("Splitter", expanded=False):
        chunk_size = st.slider(
            "Chunk Size",
            min_value=100,
            max_value=1000,
            value=DEFAULT_CHUNK_SIZE,
            step=DEFAULT_CHUNK_SIZE_STEP,
            key="settings_chunk_size",
        )
    with st.expander("Tokenization", expanded=False):
        tokenization_label = st.selectbox(
            "Tokenization Model",
            options=list(TOKENIZATION_MODEL_MAP.keys()),
            index=0,
            key="settings_tokenization_model",
        )
        

    with st.expander("Embedding", expanded=False):
        embedding_label = st.selectbox(
            "Embedding Model",
            options=list(EMBEDDING_MODEL_MAP.keys()),
            index=0,
            key="settings_embedding_model",
        )
        device_type = st.selectbox(
            "Device Type",
            options=["cpu", "cuda", "mps"],
            index=0,
            key="settings_embedding_device",
        )

    with st.expander("Retrieval", expanded=False):
        llm_label = st.selectbox(
            "LLM Model",
            options=list(LLM_MODEL_OPTIONS.keys()),
            index=0,
            key="settings_llm_model",
        )
        top_k = st.slider(
            "Top K",
            min_value=1,
            max_value=10,
            value=DEFAULT_TOP_K,
            step=1,
            key="settings_top_k",
        )

    nvidia_api_key = os.environ.get(NVIDIA_API_KEY_ENV, "").strip()
    google_api_key = os.environ.get(GOOGLE_API_KEY_ENV, "").strip()
    selected_llm = LLM_MODEL_OPTIONS[llm_label]
    llm_provider = selected_llm["provider"]
    llm_api_key = nvidia_api_key if llm_provider == NVIDIA_PROVIDER else google_api_key

    rag_settings: RagSettings = {
        "chunk_size": chunk_size,
        "tokenization_model": TOKENIZATION_MODEL_MAP[tokenization_label],
        "embedding_model": EMBEDDING_MODEL_MAP[embedding_label],
        "embedding_device": device_type,
        "llm_model": selected_llm["model"],
        "llm_display_name": llm_label,
        "llm_provider": llm_provider,
        "llm_api_key": llm_api_key,
        "top_k": top_k,
        "nvidia_api_key": nvidia_api_key,
        "google_api_key": google_api_key,
    }

    if not nvidia_api_key:
        st.warning("Set NVIDIA_API_KEY in your environment or .env file to run NVIDIA embedding models.")
    if llm_provider == GOOGLE_PROVIDER and not google_api_key:
        st.warning("Set GOOGLE_API_KEY in your environment or .env file to use gemini-2.5-flash.")

    feature_flags: dict[str, tuple[str, ...]] = {
        "reranker": ("DOCUSUN_ENABLE_RERANKER",),
        "multi query": ("DOCUSUN_ENABLE_MULTI_QUERY",),
        "parent retrieval": ("DOCUSUN_ENABLE_PARENT_RETRIEVER", "DOCUSUN_ENABLE_parent_retrieval"),
        "hybrid retrieval": ("DOCUSUN_ENABLE_HYBRID_SEARCH",),
    }

    with st.expander("Advanced Techniques", expanded=False):
        left_checks, right_checks = st.columns(2)
        with left_checks:
            reranker_enabled = st.checkbox(
                "reranker",
                value=_is_feature_enabled(feature_flags["reranker"]),
                key="retrieval_toggle_reranker",
            )
            parent_enabled = st.checkbox(
                "parent retrieval",
                value=_is_feature_enabled(feature_flags["parent retrieval"]),
                key="retrieval_toggle_parent",
            )
        with right_checks:
            multi_query_enabled = st.checkbox(
                "multi query",
                value=_is_feature_enabled(feature_flags["multi query"]),
                key="retrieval_toggle_multi_query",
            )
            hybrid_enabled = st.checkbox(
                "hybrid retrieval",
                value=_is_feature_enabled(feature_flags["hybrid retrieval"]),
                key="retrieval_toggle_hybrid",
            )

    did_update_feature_flags = _sync_retrieval_feature_flags(
        flags=feature_flags,
        values={
            "reranker": reranker_enabled,
            "multi query": multi_query_enabled,
            "parent retrieval": parent_enabled,
            "hybrid retrieval": hybrid_enabled,
        },
    )
    if did_update_feature_flags:
        st.caption("Technique flags updated in .env")

    has_uploaded_file = uploaded_file is not None
    has_cached_index = False
    if has_uploaded_file:
        cached_file_key = _build_file_key(uploaded_file.name, uploaded_file.getvalue(), rag_settings)
        has_cached_index = cached_file_key in _get_file_cache()

    processing_expander_label = "Processing steps"
    if has_uploaded_file:
        if has_cached_index:
            processing_expander_label = "processing done"
        else:
            processing_expander_label = "⏳ processing the document"

    st.markdown("### Processing")
    processing_expander = st.expander(processing_expander_label, expanded=has_uploaded_file)
    with processing_expander:
        processing_steps_container = st.container()
        if uploaded_file is None:
            processing_steps_container.markdown(":gray[No processing has started yet.]")

    active_doc: Optional[CachedDocument] = None

    if uploaded_file is not None:
        step_log_fn, finalize_step_log = _build_index_step_logger(processing_steps_container)
        did_index_new_file = not has_cached_index
        try:
            if not rag_settings["nvidia_api_key"]:
                raise ValueError("NVIDIA_API_KEY is required for NVIDIA Build API models.")
            active_doc = _prepare_document(
                uploaded_file,
                settings=rag_settings,
                log_fn=step_log_fn,
            )
            finalize_step_log()
        except Exception as exc:
            finalize_step_log()
            st.markdown(":red[✗ failed while processing document]")
            st.exception(exc)
            st.stop()

        if did_index_new_file:
            st.rerun()

        st.caption(active_doc["file_name"])
        display_pdf(active_doc["file_bytes"])


header_col, clear_col = st.columns([7, 1], vertical_alignment="bottom")
with header_col:
    st.header("Chat with Your Documents")
with clear_col:
    st.button("Clear ↺", key="clear_chat_btn", on_click=reset_chat)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            elapsed = message.get("query_time_seconds")
            elapsed_value = float(elapsed) if isinstance(elapsed, (int, float)) else None
            model_hint = message.get("llm_model_hint")
            _render_assistant_content(
                content=str(message["content"]),
                elapsed_seconds=elapsed_value,
                model_hint=str(model_hint) if isinstance(model_hint, str) and model_hint else None,
            )
        else:
            st.markdown(message["content"])


if prompt := st.chat_input("What's on your mind?"):
    if uploaded_file is None:
        st.error("Please upload a PDF first.")
        st.stop()

    active_key = st.session_state.active_file_key
    file_cache = _get_file_cache()
    if not isinstance(active_key, str) or active_key not in file_cache:
        st.error("Document index is not ready yet. Please re-upload the file.")
        st.stop()

    active_doc = file_cache[active_key]

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_box = st.empty()
        query_elapsed_seconds = 0.0
        try:
            if not rag_settings["llm_api_key"]:
                required_key_name = GOOGLE_API_KEY_ENV if rag_settings["llm_provider"] == GOOGLE_PROVIDER else NVIDIA_API_KEY_ENV
                raise ValueError(f"{required_key_name} is required for the selected LLM model.")
            start_time = time.perf_counter()
            answer = _answer_question(
                question=prompt,
                persist_dir=active_doc["persist_dir"],
                settings=rag_settings,
            )
            query_elapsed_seconds = time.perf_counter() - start_time
        except Exception as exc:
            st.exception(exc)
            st.stop()

        streamed = ""
        for token in _stream_words(str(answer)):
            streamed += token
            _render_assistant_content(
                content=streamed,
                elapsed_seconds=query_elapsed_seconds,
                target=response_box,
                with_cursor=True,
            )
        _render_assistant_content(
            content=streamed,
            elapsed_seconds=query_elapsed_seconds,
            model_hint=rag_settings["llm_display_name"],
            target=response_box,
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": str(answer),
            "query_time_seconds": query_elapsed_seconds,
            "llm_model_hint": rag_settings["llm_display_name"],
        }
    )
