from typing import Iterable


def write_header(out, title: str) -> None:
    out.write("\n" + "*" * 70 + "\n")
    out.write(title + "\n")
    out.write("*" * 70 + "\n")


def write_kv(out, key: str, value) -> None:
    out.write(f"{key}: {value}\n")


def write_sources(out, paths: Iterable[str]) -> None:
    for path in paths:
        out.write(f"{path}\n")


def write_documents(out, documents) -> None:
    for i, doc in enumerate(documents):
        out.write(f"\n--- Document #{i} ---\n")
        out.write(f"metadata: {doc.metadata}\n")
        out.write(doc.page_content + "\n")


def write_chunks(out, chunks) -> None:
    for i, chunk in enumerate(chunks):
        out.write(f"\n# chunk {i}\n")
        out.write(f"metadata: {chunk.metadata}\n")
        out.write(chunk.page_content + "\n")


def write_similar_chunks(out, chunks) -> None:
    for i, chunk in enumerate(chunks):
        out.write(f"\n--- chunk {i} ---\n")
        out.write(f"metadata: {chunk.metadata}\n")
        out.write(chunk.page_content + "\n")


def write_question_answer(out, question: str, answer: str) -> None:
    out.write(f"Question: {question}\n")
    out.write(f"Answer: {answer}\n")
