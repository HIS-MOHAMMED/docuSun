import base64
import gc
import hashlib
import os
import re
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Optional, TypedDict, cast

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
    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_display = (
        f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
        'width="100%" height="900" type="application/pdf" '
        'style="border:none; border-radius:8px;">'
        "</iframe>"
    )
    st.markdown(pdf_display, unsafe_allow_html=True)


def _build_file_key(filename: str, content: bytes, chunk_size: int, top_k: int) -> str:
    digest = hashlib.sha1(content).hexdigest()  # nosec B324
    return f"{filename}:{digest}:chunk{chunk_size}:topk{top_k}"


def _stream_words(text: str):
    words = text.split(" ")
    for i, word in enumerate(words):
        suffix = " " if i < len(words) - 1 else ""
        yield f"{word}{suffix}"


def _prepare_document(uploaded_file, chunk_size: int, top_k: int) -> CachedDocument:
    file_bytes = uploaded_file.getvalue()
    file_key = _build_file_key(uploaded_file.name, file_bytes, chunk_size, top_k)
    file_cache = _get_file_cache()

    if file_key in file_cache:
        st.session_state.active_file_key = file_key
        return file_cache[file_key]

    workdir = tempfile.mkdtemp(prefix=f"docusun_ui_{st.session_state.session_id}_")
    data_dir = os.path.join(workdir, "data")
    persist_dir = os.path.join(workdir, "chroma")
    os.makedirs(data_dir, exist_ok=True)

    pdf_path = os.path.join(data_dir, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(file_bytes)

    try:
        from engine.qa.pipeline import index_documents

        index_documents(
            data_path=data_dir,
            chunk_size=chunk_size,
            top_k=top_k,
            persist_directory=persist_dir,
        )
    except Exception:
        raise

    payload: CachedDocument = {
        "file_name": uploaded_file.name,
        "file_key": file_key,
        "persist_dir": persist_dir,
        "workdir": workdir,
        "file_bytes": file_bytes,
    }
    file_cache[file_key] = payload
    st.session_state.active_file_key = file_key
    return payload


def _answer_question(question: str, persist_dir: str, top_k: int) -> str:
    from engine.qa.pipeline import query_documents

    return query_documents(
        question=question,
        top_k=top_k,
        persist_directory=persist_dir,
    )


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### Upload Document")
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type=["pdf"])

    top_k = st.slider("Top K", min_value=1, max_value=10, value=3, step=1)
    chunk_size = st.slider("Chunk Size", min_value=100, max_value=1200, value=400, step=50)
    feature_flags: dict[str, tuple[str, ...]] = {
        "reranker": ("DOCUSUN_ENABLE_RERANKER",),
        "multi query": ("DOCUSUN_ENABLE_MULTI_QUERY",),
        "parent retrieval": ("DOCUSUN_ENABLE_PARENT_RETRIEVER", "DOCUSUN_ENABLE_parent_retrieval"),
        "hybrid retrieval": ("DOCUSUN_ENABLE_HYBRID_SEARCH",),
    }

    st.markdown("### Advanced Techniques")
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

    active_doc: Optional[CachedDocument] = None

    if uploaded_file is not None:
        with st.status("processing your document", expanded=False, state="running") as status:
            try:
                active_doc = _prepare_document(uploaded_file, chunk_size=chunk_size, top_k=top_k)
            except Exception as exc:
                status.update(state="error", label="failed while processing document")
                st.exception(exc)
                st.stop()

            status.update(state="complete", label="processing complete")

        st.caption(active_doc["file_name"])
        display_pdf(active_doc["file_bytes"])


header_col, clear_col = st.columns([7, 1], vertical_alignment="bottom")
with header_col:
    st.header("Chat with Your Documents")
with clear_col:
    st.button("Clear ↺", key="clear_chat_btn", on_click=reset_chat)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
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
        try:
            answer = _answer_question(
                question=prompt,
                persist_dir=active_doc["persist_dir"],
                top_k=top_k,
            )
        except Exception as exc:
            st.exception(exc)
            st.stop()

        streamed = ""
        for token in _stream_words(str(answer)):
            streamed += token
            response_box.markdown(streamed + "▌")
        response_box.markdown(streamed)

    st.session_state.messages.append({"role": "assistant", "content": str(answer)})
