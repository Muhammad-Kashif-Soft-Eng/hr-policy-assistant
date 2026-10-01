import hashlib
import io
import json
import os
import re

import faiss
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


MODEL_NAME = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
CHUNK_SIZE = 220
CHUNK_OVERLAP = 40


st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📘",
    layout="wide",
)


def get_groq_api_key():
    """Read the Groq key from Streamlit Secrets first, then environment variables."""
    try:
        secret_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        secret_key = ""

    return secret_key or os.getenv("GROQ_API_KEY", "")


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def normalize_text(text):
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_page_text(text, page_number):
    """Create overlapping word chunks while preserving the PDF page number."""
    words = normalize_text(text).split()

    if not words:
        return []

    chunks = []
    step = max(CHUNK_SIZE - CHUNK_OVERLAP, 1)

    for start in range(0, len(words), step):
        end = min(start + CHUNK_SIZE, len(words))
        chunk_text = " ".join(words[start:end]).strip()

        if chunk_text:
            chunks.append(
                {
                    "page": page_number,
                    "text": chunk_text,
                }
            )

        if end >= len(words):
            break

    return chunks


def extract_pdf_chunks(pdf_bytes):
    """Extract text page-by-page and convert it into retrieval chunks."""
    chunks = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        for page_index, page in enumerate(document, start=1):
            page_text = page.get_text("text")
            chunks.extend(chunk_page_text(page_text, page_index))

    return chunks


def build_faiss_index(chunks, model):
    texts = [item["text"] for item in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


def retrieve_chunks(question, index, chunks, model, top_k=TOP_K):
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_embedding = np.asarray(query_embedding, dtype="float32")

    scores, positions = index.search(
        query_embedding,
        min(top_k, len(chunks)),
    )

    results = []

    for score, position in zip(scores[0], positions[0]):
        if position < 0:
            continue

        item = chunks[int(position)].copy()
        item["score"] = float(score)
        results.append(item)

    return results


def build_context(retrieved_chunks):
    context_parts = []

    for i, item in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {i} | PDF page {item['page']} | similarity {item['score']:.3f}]\n"
            f"{item['text']}"
        )

    return "\n\n".join(context_parts)


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["ANSWERED", "NOT_FOUND"],
        },
        "answer": {
            "type": "string",
        },
        "key_points": {
            "type": "array",
            "items": {"type": "string"},
        },
        "policy_basis": {
            "type": "array",
            "items": {"type": "string"},
        },
        "caveats": {
            "type": "array",
            "items": {"type": "string"},
        },
        "confidence": {
            "type": "string",
            "enum": ["High", "Medium", "Low"],
        },
    },
    "required": [
        "status",
        "answer",
        "key_points",
        "policy_basis",
        "caveats",
        "confidence",
    ],
    "additionalProperties": False,
}


def ask_groq(question, context, api_key):
    client = Groq(api_key=api_key)

    system_prompt = """
You are an HR Policy Assistant. Answer questions using ONLY the supplied
retrieved excerpts from the uploaded HR policy PDF.

Grounding rules:
1. Never invent a policy, rule, number, deadline, eligibility condition, benefit,
   or exception that is not supported by the excerpts.
2. If the excerpts do not contain enough information, return status NOT_FOUND,
   clearly say that the uploaded policy does not provide enough information,
   and do not guess.
3. Prefer precise, plain language suitable for an employee or HR administrator.
4. Policy_basis must summarize the specific policy statements that support the
   answer. Do not cite sources that are not present in the context.
5. Caveats should mention genuine limits or missing details only.
6. Do not provide legal advice. When legal interpretation would be needed,
   say the policy text alone is insufficient.
7. Keep the final answer concise but useful.
"""

    user_prompt = f"""
Question:
{question}

Retrieved policy excerpts:
{context}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0,
        reasoning_effort="low",
        max_completion_tokens=1000,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "hr_policy_answer",
                "strict": True,
                "schema": ANSWER_SCHEMA,
            },
        },
    )

    content = response.choices[0].message.content or "{}"
    return json.loads(content)


def show_structured_answer(answer, retrieved_chunks):
    status = answer.get("status", "ANSWERED")

    if status == "ANSWERED":
        st.success("Policy answer found in the retrieved document context.")
    else:
        st.warning(
            "The retrieved policy context does not contain enough information "
            "to answer this question safely."
        )

    st.subheader("Answer")
    st.write(answer.get("answer", ""))

    left, right = st.columns(2)

    with left:
        st.markdown("#### Key Points")
        points = answer.get("key_points", [])
        if points:
            for point in points:
                st.markdown(f"- {point}")
        else:
            st.write("No additional key points.")

    with right:
        st.markdown("#### Confidence")
        st.info(answer.get("confidence", "Low"))

    st.markdown("#### Policy Basis")
    basis = answer.get("policy_basis", [])
    if basis:
        for item in basis:
            st.markdown(f"- {item}")
    else:
        st.write("No supporting policy basis was identified.")

    st.markdown("#### Caveats")
    caveats = answer.get("caveats", [])
    if caveats:
        for item in caveats:
            st.markdown(f"- {item}")
    else:
        st.write("No additional caveats.")

    st.markdown("#### Retrieved Sources")
    source_rows = [
        {
            "Source": f"Source {i}",
            "PDF Page": item["page"],
            "Similarity": round(item["score"], 3),
            "Excerpt": item["text"][:280] + ("…" if len(item["text"]) > 280 else ""),
        }
        for i, item in enumerate(retrieved_chunks, start=1)
    ]

    st.dataframe(
        source_rows,
        use_container_width=True,
        hide_index=True,
    )


st.title("📘 HR Policy Assistant")
st.caption(
    "Upload an HR policy PDF and ask questions. "
    "Answers are grounded in the uploaded document using RAG."
)

with st.sidebar:
    st.header("Document")
    uploaded_file = st.file_uploader(
        "Upload one HR Policy PDF",
        type=["pdf"],
        accept_multiple_files=False,
    )

    top_k = st.slider(
        "Retrieved passages",
        min_value=3,
        max_value=8,
        value=TOP_K,
        step=1,
    )

    st.divider()
    st.markdown("**RAG stack**")
    st.caption(
        "PyMuPDF → Sentence Transformers → FAISS → Groq GPT-OSS 20B"
    )

    if st.button("Clear current document", use_container_width=True):
        for key in [
            "document_key",
            "document_name",
            "chunks",
            "faiss_index",
            "last_answer",
            "last_sources",
        ]:
            st.session_state.pop(key, None)
        st.rerun()


if uploaded_file is None:
    st.info(
        "Upload an HR Policy PDF from the sidebar to build the searchable "
        "knowledge base."
    )
    st.markdown(
        """
### What this app does

1. Extracts text from the PDF with **PyMuPDF**.
2. Splits the document into overlapping, page-aware chunks.
3. Creates semantic embeddings with **Sentence Transformers**.
4. Stores embeddings in a **FAISS** similarity index.
5. Retrieves the most relevant policy passages for each question.
6. Sends only those passages to **Groq `openai/gpt-oss-20b`**.
7. Displays a structured, grounded response with source pages.
        """
    )
    st.stop()


pdf_bytes = uploaded_file.getvalue()
document_key = hashlib.sha256(pdf_bytes).hexdigest()

if st.session_state.get("document_key") != document_key:
    with st.spinner("Reading the HR policy and building the searchable index..."):
        try:
            embedding_model = load_embedding_model()
            chunks = extract_pdf_chunks(pdf_bytes)

            if not chunks:
                st.error(
                    "No readable text was found in this PDF. "
                    "Please upload a text-based HR policy PDF."
                )
                st.stop()

            index = build_faiss_index(chunks, embedding_model)

            st.session_state["document_key"] = document_key
            st.session_state["document_name"] = uploaded_file.name
            st.session_state["chunks"] = chunks
            st.session_state["faiss_index"] = index
            st.session_state.pop("last_answer", None)
            st.session_state.pop("last_sources", None)

        except Exception as exc:
            st.error(f"Could not process this PDF: {exc}")
            st.stop()


chunks = st.session_state["chunks"]
faiss_index = st.session_state["faiss_index"]
embedding_model = load_embedding_model()

st.success(
    f"**{st.session_state['document_name']}** is ready — "
    f"{len(chunks)} searchable passages created."
)

question = st.text_input(
    "Ask a question about the uploaded HR policy",
    placeholder="Example: How many annual leave days are employees entitled to?",
)

ask_clicked = st.button(
    "Ask HR Policy Assistant",
    type="primary",
    use_container_width=True,
)

if ask_clicked:
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    api_key = get_groq_api_key()

    if not api_key:
        st.error(
            "GROQ_API_KEY is missing. Add it in Streamlit Community Cloud "
            "under App Settings → Secrets."
        )
        st.stop()

    with st.spinner("Retrieving relevant policy sections and preparing the answer..."):
        try:
            retrieved = retrieve_chunks(
                question.strip(),
                faiss_index,
                chunks,
                embedding_model,
                top_k=top_k,
            )

            context = build_context(retrieved)
            answer = ask_groq(question.strip(), context, api_key)

            st.session_state["last_answer"] = answer
            st.session_state["last_sources"] = retrieved

        except Exception as exc:
            st.error(f"The question could not be answered: {exc}")
            st.stop()


if "last_answer" in st.session_state:
    st.divider()
    show_structured_answer(
        st.session_state["last_answer"],
        st.session_state.get("last_sources", []),
    )

st.divider()
st.caption(
    "Grounding note: This assistant answers from the uploaded policy text only. "
    "For legal or case-specific HR decisions, verify the official policy and "
    "appropriate HR/legal guidance."
)
