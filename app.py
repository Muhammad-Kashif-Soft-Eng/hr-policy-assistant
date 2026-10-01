import hashlib
import json
import os
import re

import numpy as np
import streamlit as st


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_NAME = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_TOP_K = 5
CHUNK_SIZE = 220
CHUNK_OVERLAP = 40


st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📘",
    layout="wide",
)


# =========================================================
# DEPENDENCY LOADERS
# =========================================================

@st.cache_resource
def load_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            EMBEDDING_MODEL
        )

    except Exception as exc:
        st.error(
            "Sentence Transformers could not be loaded."
        )

        st.code(
            str(exc),
            language="text"
        )

        st.info(
            "Check requirements.txt and make sure "
            "sentence-transformers is installed."
        )

        raise


@st.cache_resource
def load_faiss():
    try:
        import faiss

        return faiss

    except Exception as exc:
        st.error(
            "FAISS could not be loaded."
        )

        st.code(
            str(exc),
            language="text"
        )

        st.info(
            "Check requirements.txt and make sure "
            "faiss-cpu is installed."
        )

        raise


def load_groq_class():
    try:
        from groq import Groq

        return Groq

    except Exception as exc:
        st.error(
            "The Groq Python SDK could not be loaded."
        )

        st.code(
            str(exc),
            language="text"
        )

        st.info(
            "Check requirements.txt and make sure "
            "groq is installed."
        )

        raise


# =========================================================
# GROQ API KEY
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
        or os.getenv(
            "GROQ_API_KEY",
            ""
        )
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
# PDF PROCESSING
# =========================================================

def extract_pdf_chunks(
    pdf_bytes
):

    import pymupdf

    chunks = []

    with pymupdf.open(
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
    embedding_model
):

    faiss = load_faiss()

    texts = [
        item["text"]
        for item in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
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
    embedding_model,
    top_k
):

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
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
# CONTEXT CREATION
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
            f"PDF Page {item['page']} | "
            f"Similarity {item['score']:.3f}]\n"
            f"{item['text']}"
        )

    return "\n\n".join(
        context_parts
    )


# =========================================================
# STRUCTURED OUTPUT SCHEMA
# =========================================================

ANSWER_SCHEMA = {

    "type": "object",

    "properties": {

        "status": {
            "type": "string",
            "enum": [
                "ANSWERED",
                "NOT_FOUND"
            ]
        },

        "answer": {
            "type": "string"
        },

        "key_points": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },

        "policy_basis": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },

        "caveats": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },

        "confidence": {
            "type": "string",
            "enum": [
                "High",
                "Medium",
                "Low"
            ]
        }
    },

    "required": [
        "status",
        "answer",
        "key_points",
        "policy_basis",
        "caveats",
        "confidence"
    ],

    "additionalProperties": False
}


# =========================================================
# GROQ QUERY
# =========================================================

def ask_groq(
    question,
    context,
    api_key
):

    Groq = load_groq_class()

    client = Groq(
        api_key=api_key
    )

    system_prompt = """
You are an HR Policy Assistant.

Your task is to answer questions ONLY from
the retrieved excerpts of the uploaded HR Policy PDF.

GROUNDING RULES:

1. Never invent a policy.
2. Never invent a number, date, leave balance,
   eligibility rule, benefit, deadline, procedure,
   exception, or requirement.
3. Use only information contained in the
   retrieved policy excerpts.
4. If the excerpts do not contain enough
   information, return status = NOT_FOUND.
5. Do not guess.
6. Keep the answer clear and practical.
7. Explain the policy basis supporting the answer.
8. Mention meaningful caveats when necessary.
9. Do not provide legal advice.
10. Do not use outside knowledge to fill missing policy details.

RESPONSE STYLE:

Answer directly.
Use simple professional language.
Keep the response concise but useful.
"""

    user_prompt = f"""
USER QUESTION:

{question}

RETRIEVED HR POLICY EXCERPTS:

{context}
"""

    response = client.chat.completions.create(

        model=MODEL_NAME,

        messages=[
            {
                "role": "system",
                "content": system_prompt.strip()
            },
            {
                "role": "user",
                "content": user_prompt.strip()
            }
        ],

        temperature=0,

        reasoning_effort="low",

        max_completion_tokens=1000,

        response_format={
            "type": "json_schema",

            "json_schema": {
                "name": "hr_policy_answer",
                "strict": True,
                "schema": ANSWER_SCHEMA
            }
        }
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
# STRUCTURED ANSWER UI
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
            "the retrieved HR policy content."
        )

    else:

        st.warning(
            "The uploaded HR policy does not "
            "contain enough information to answer "
            "this question."
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
    # KEY POINTS
    # -----------------------------------------------------

    left_column, right_column = st.columns(2)

    with left_column:

        st.markdown(
            "#### Key Points"
        )

        key_points = answer.get(
            "key_points",
            []
        )

        if key_points:

            for point in key_points:

                st.markdown(
                    f"- {point}"
                )

        else:

            st.write(
                "No additional key points."
            )

    # -----------------------------------------------------
    # CONFIDENCE
    # -----------------------------------------------------

    with right_column:

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
            "No supporting policy basis was identified."
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
    # RETRIEVED SOURCES
    # -----------------------------------------------------

    st.markdown(
        "#### Retrieved Sources"
    )

    source_rows = []

    for number, item in enumerate(
        retrieved_chunks,
        start=1
    ):

        excerpt = item["text"]

        if len(excerpt) > 300:
            excerpt = excerpt[:300] + "..."

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
                    excerpt
            }
        )

    if source_rows:

        st.dataframe(
            source_rows,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.write(
            "No source passages were retrieved."
        )


# =========================================================
# HEADER
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
        accept_multiple_files=False
    )

    top_k = st.slider(
        "Retrieved passages",
        min_value=3,
        max_value=8,
        value=DEFAULT_TOP_K,
        step=1
    )

    st.divider()

    st.markdown(
        "**RAG Pipeline**"
    )

    st.caption(
        "PyMuPDF → Sentence Transformers → "
        "FAISS → Groq GPT-OSS 20B"
    )

    st.divider()

    st.markdown(
        "**Response Structure**"
    )

    st.caption(
        "Answer → Key Points → Policy Basis → "
        "Caveats → Confidence → Sources"
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
            "last_sources"
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
        "Upload an HR Policy PDF from "
        "the sidebar to begin."
    )

    st.markdown(
        """
### How it works

1. PyMuPDF extracts text from the PDF.
2. The text is split into overlapping chunks.
3. Sentence Transformers creates semantic embeddings.
4. FAISS stores and searches the embeddings.
5. Relevant policy passages are retrieved.
6. Groq GPT-OSS 20B generates a grounded response.
7. The answer is shown in a structured format.
"""
    )

    st.stop()


# =========================================================
# DOCUMENT PROCESSING
# =========================================================

pdf_bytes = uploaded_file.getvalue()

document_key = hashlib.sha256(
    pdf_bytes
).hexdigest()


if st.session_state.get(
    "document_key"
) != document_key:

    with st.spinner(
        "Reading the HR Policy PDF and building the search index..."
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
                    "in this PDF. Please upload a "
                    "text-based HR policy PDF."
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
                "The PDF could not be processed."
            )

            st.code(
                str(exc),
                language="text"
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
    )
)


ask_clicked = st.button(
    "Ask HR Policy Assistant",
    type="primary",
    use_container_width=True
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
            "GROQ_API_KEY is missing."
        )

        st.info(
            "Go to Streamlit Cloud → "
            "Manage app → Settings → Secrets "
            "and add your Groq API key."
        )

        st.stop()

    with st.spinner(
        "Searching the HR policy and generating the answer..."
    ):

        try:

            retrieved_chunks = retrieve_chunks(
                question.strip(),
                faiss_index,
                chunks,
                embedding_model,
                top_k
            )

            if not retrieved_chunks:

                st.warning(
                    "No relevant policy passages were found."
                )

                st.stop()

            context = build_context(
                retrieved_chunks
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
            ] = retrieved_chunks

        except Exception as exc:

            st.error(
                "The question could not be answered."
            )

            st.code(
                str(exc),
                language="text"
            )

            st.stop()


# =========================================================
# DISPLAY RESULT
# =========================================================

if "last_answer" in st.session_state:

    st.divider()

    show_structured_answer(
        st.session_state[
            "last_answer"
        ],
        st.session_state.get(
            "last_sources",
            []
        )
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "This assistant answers from the uploaded "
    "policy text only. Verify important HR or "
    "legal decisions against the official policy "
    "and appropriate HR guidance."
)
