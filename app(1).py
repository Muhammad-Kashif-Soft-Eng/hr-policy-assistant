# HR Policy Assistant

## 1. app.py

import hashlib
import json
import os
import re

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer


MODEL_NAME = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5
CHUNK_SIZE = 220
CHUNK_OVERLAP = 40


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📘",
    layout="wide",
)


# =========================================================
# FAISS LOADER
# =========================================================

@st.cache_resource
def load_faiss():

    try:

        import faiss

        return faiss

    except ImportError as exc:

        st.error(
            """
            FAISS is not installed correctly.

            Please check:

            1. requirements.txt contains:
               faiss-cpu==1.10.0

            2. Streamlit Cloud is using Python 3.12.

            3. requirements.txt is in the repository root.

            4. You redeployed the application after
               changing requirements.txt.
            """
        )

        raise exc


# =========================================================
# GROQ LOADER
# =========================================================

@st.cache_resource
def load_groq():

    try:

        from groq import Groq

        return Groq

    except ImportError as exc:

        st.error(
            """
            The Groq Python package is not installed.

            Make sure requirements.txt contains:

            groq==1.7.0

            Then commit the updated requirements.txt
            to GitHub and redeploy the Streamlit app.
            """
        )

        raise exc


# =========================================================
# API KEY
# =========================================================

def get_groq_api_key():

    try:

        secret_key = st.secrets.get(
            "GROQ_API_KEY",
            ""
        )

    except Exception:

        secret_key = ""

    return (
        secret_key
        or os.getenv("GROQ_API_KEY", "")
    )


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text):

    text = text.replace(
        "\u00a0",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# TEXT CHUNKING
# =========================================================

def chunk_page_text(
    text,
    page_number
):

    words = normalize_text(
        text
    ).split()

    if not words:

        return []

    chunks = []

    step = max(
        CHUNK_SIZE - CHUNK_OVERLAP,
        1
    )

    for start in range(
        0,
        len(words),
        step
    ):

        end = min(
            start + CHUNK_SIZE,
            len(words)
        )

        chunk_text = " ".join(
            words[start:end]
        ).strip()

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


# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_pdf_chunks(
    pdf_bytes
):

    import fitz

    chunks = []

    with fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    ) as document:

        for page_number, page in enumerate(
            document,
            start=1
        ):

            page_text = page.get_text(
                "text"
            )

            page_chunks = chunk_page_text(
                page_text,
                page_number
            )

            chunks.extend(
                page_chunks
            )

    return chunks


# =========================================================
# FAISS INDEX
# =========================================================

def build_faiss_index(
    chunks,
    model
):

    faiss = load_faiss()

    texts = [
        item["text"]
        for item in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(
        embeddings
    )

    return index


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_chunks(
    question,
    index,
    chunks,
    model,
    top_k
):

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, positions = index.search(
        query_embedding,
        min(
            top_k,
            len(chunks)
        )
    )

    results = []

    for score, position in zip(
        scores[0],
        positions[0]
    ):

        if position < 0:

            continue

        item = chunks[
            int(position)
        ].copy()

        item["score"] = float(
            score
        )

        results.append(
            item
        )

    return results


# =========================================================
# BUILD CONTEXT
# =========================================================

def build_context(
    retrieved_chunks
):

    context_parts = []

    for number, item in enumerate(
        retrieved_chunks,
        start=1
    ):

        context_parts.append(
            f"[Source {number} | "
            f"PDF page {item['page']} | "
            f"similarity {item['score']:.3f}]\n"
            f"{item['text']}"
        )

    return "\n\n".join(
        context_parts
    )


# =========================================================
# STRUCTURED RESPONSE SCHEMA
# =========================================================

ANSWER_SCHEMA = {

    "type": "object",

    "properties": {

        "status": {
            "type": "string",
            "enum": [
                "ANSWERED",
                "NOT_FOUND"
            ],
        },

        "answer": {
            "type": "string",
        },

        "key_points": {
            "type": "array",
            "items": {
                "type": "string"
            },
        },

        "policy_basis": {
            "type": "array",
            "items": {
                "type": "string"
            },
        },

        "caveats": {
            "type": "array",
            "items": {
                "type": "string"
            },
        },

        "confidence": {
            "type": "string",
            "enum": [
                "High",
                "Medium",
                "Low"
            ],
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


# =========================================================
# ASK GROQ
# =========================================================

def ask_groq(
    question,
    context,
    api_key
):

    Groq = load_groq()

    client = Groq(
        api_key=api_key
    )

    system_prompt = """
You are an HR Policy Assistant.

Answer questions ONLY from the retrieved
excerpts of the uploaded HR Policy PDF.

Rules:

1. Never invent HR policies.
2. Never invent numbers, dates,
   leave balances, benefits,
   eligibility rules or deadlines.
3. If the retrieved context does not contain
   enough information, return NOT_FOUND.
4. Give a clear direct answer.
5. Explain which policy information
   supports the answer.
6. Mention meaningful caveats.
7. Do not provide legal advice.
8. Do not use information outside
   the provided policy excerpts.
9. Keep the response concise.
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
            {
                "role": "system",
                "content": system_prompt.strip(),
            },
            {
                "role": "user",
                "content": user_prompt.strip(),
            },
        ],

        temperature=0,

        reasoning_effort="low",

        max_completion_tokens=1000,

        response_format={

            "type": "json_schema",

            "json_schema": {

                "name":
                    "hr_policy_answer",

                "strict":
                    True,

                "schema":
                    ANSWER_SCHEMA,

            },

        },

    )

    content = (
        response
        .choices[0]
        .message
        .content
        or "{}"
    )

    return json.loads(
        content
    )


# =========================================================
# DISPLAY STRUCTURED ANSWER
# =========================================================

def show_structured_answer(
    answer,
    retrieved_chunks
):

    status = answer.get(
        "status",
        "ANSWERED"
    )

    if status == "ANSWERED":

        st.success(
            "Answer generated from "
            "the retrieved HR policy context."
        )

    else:

        st.warning(
            "The uploaded policy does not contain "
            "enough information to answer this question."
        )

    # -----------------------------------------------------
    # ANSWER
    # -----------------------------------------------------

    st.subheader(
        "Answer"
    )

    st.write(
        answer.get(
            "answer",
            ""
        )
    )

    # -----------------------------------------------------
    # KEY POINTS / CONFIDENCE
    # -----------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.markdown(
            "#### Key Points"
        )

        points = answer.get(
            "key_points",
            []
        )

        if points:

            for point in points:

                st.markdown(
                    f"- {point}"
                )

        else:

            st.write(
                "No additional key points."
            )

    with right:

        st.markdown(
            "#### Confidence"
        )

        st.info(
            answer.get(
                "confidence",
                "Low"
            )
        )

    # -----------------------------------------------------
    # POLICY BASIS
    # -----------------------------------------------------

    st.markdown(
        "#### Policy Basis"
    )

    policy_basis = answer.get(
        "policy_basis",
        []
    )

    if policy_basis:

        for item in policy_basis:

            st.markdown(
                f"- {item}"
            )

    else:

        st.write(
            "No policy basis was identified."
        )

    # -----------------------------------------------------
    # CAVEATS
    # -----------------------------------------------------

    st.markdown(
        "#### Caveats"
    )

    caveats = answer.get(
        "caveats",
        []
    )

    if caveats:

        for item in caveats:

            st.markdown(
                f"- {item}"
            )

    else:

        st.write(
            "No additional caveats."
        )

    # -----------------------------------------------------
    # SOURCES
    # -----------------------------------------------------

    st.markdown(
        "#### Retrieved Sources"
    )

    source_rows = []

    for number, item in enumerate(
        retrieved_chunks,
        start=1
    ):

        source_rows.append(
            {
                "Source":
                    f"Source {number}",

                "PDF Page":
                    item["page"],

                "Similarity":
                    round(
                        item["score"],
                        3
                    ),

                "Excerpt":
                    item["text"][:300]
                    + (
                        "..."
                        if len(
                            item["text"]
                        ) > 300
                        else ""
                    ),
            }
        )

    st.dataframe(
        source_rows,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# MAIN UI
# =========================================================

st.title(
    "📘 HR Policy Assistant"
)

st.caption(
    "Upload an HR Policy PDF and ask "
    "questions using Retrieval-Augmented Generation."
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header(
        "Document"
    )

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

    st.markdown(
        "**RAG Pipeline**"
    )

    st.caption(
        "PyMuPDF → Sentence Transformers → "
        "FAISS → Groq GPT-OSS 20B"
    )

    if st.button(
        "Clear Current Document",
        use_container_width=True
    ):

        for key in [
            "document_key",
            "document_name",
            "chunks",
            "faiss_index",
            "last_answer",
            "last_sources",
        ]:

            st.session_state.pop(
                key,
                None
            )

        st.rerun()


# =========================================================
# EMPTY STATE
# =========================================================

if uploaded_file is None:

    st.info(
        "Upload an HR Policy PDF "
        "from the sidebar."
    )

    st.markdown(
        """
### How It Works

1. Extract PDF text with PyMuPDF.
2. Split text into overlapping chunks.
3. Create semantic embeddings.
4. Store embeddings in FAISS.
5. Retrieve relevant policy sections.
6. Send retrieved context to Groq.
7. Display the structured answer.
"""
    )

    st.stop()


# =========================================================
# PROCESS PDF
# =========================================================

pdf_bytes = uploaded_file.getvalue()

document_key = hashlib.sha256(
    pdf_bytes
).hexdigest()


if (
    st.session_state.get(
        "document_key"
    )
    != document_key
):

    with st.spinner(
        "Processing HR Policy PDF..."
    ):

        try:

            embedding_model = (
                load_embedding_model()
            )

            chunks = extract_pdf_chunks(
                pdf_bytes
            )

            if not chunks:

                st.error(
                    "No readable text was found "
                    "in this PDF."
                )

                st.stop()

            index = build_faiss_index(
                chunks,
                embedding_model
            )

            st.session_state[
                "document_key"
            ] = document_key

            st.session_state[
                "document_name"
            ] = uploaded_file.name

            st.session_state[
                "chunks"
            ] = chunks

            st.session_state[
                "faiss_index"
            ] = index

            st.session_state.pop(
                "last_answer",
                None
            )

            st.session_state.pop(
                "last_sources",
                None
            )

        except Exception as exc:

            st.error(
                f"Could not process the PDF: {exc}"
            )

            st.stop()


# =========================================================
# DOCUMENT READY
# =========================================================

chunks = st.session_state[
    "chunks"
]

faiss_index = st.session_state[
    "faiss_index"
]

embedding_model = (
    load_embedding_model()
)


st.success(
    f"**{st.session_state['document_name']}** "
    f"is ready — {len(chunks)} searchable "
    f"passages created."
)


# =========================================================
# QUESTION
# =========================================================

question = st.text_input(
    "Ask a question about the uploaded HR policy",
    placeholder=(
        "Example: How many annual leave "
        "days are employees entitled to?"
    ),
)


ask_clicked = st.button(
    "Ask HR Policy Assistant",
    type="primary",
    use_container_width=True,
)


# =========================================================
# QUESTION PROCESSING
# =========================================================

if ask_clicked:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()

    api_key = get_groq_api_key()

    if not api_key:

        st.error(
            "GROQ_API_KEY is missing. "
            "Add it in Streamlit Cloud → "
            "Settings → Secrets."
        )

        st.stop()

    with st.spinner(
        "Searching the policy and generating the answer..."
    ):

        try:

            retrieved = retrieve_chunks(
                question.strip(),
                faiss_index,
                chunks,
                embedding_model,
                top_k
            )

            context = build_context(
                retrieved
            )

            answer = ask_groq(
                question.strip(),
                context,
                api_key
            )

            st.session_state[
                "last_answer"
            ] = answer

            st.session_state[
                "last_sources"
            ] = retrieved

        except Exception as exc:

            st.error(
                f"The question could not be answered: {exc}"
            )

            st.stop()


# =========================================================
# SHOW RESULT
# =========================================================

if (
    "last_answer"
    in st.session_state
):

    st.divider()

    show_structured_answer(
        st.session_state[
            "last_answer"
        ],
        st.session_state.get(
            "last_sources",
            []
        ),
    )


st.divider()

st.caption(
    "This assistant answers from the uploaded "
    "policy text only. Verify important HR "
    "or legal decisions against the official "
    "policy and appropriate HR guidance."
)
