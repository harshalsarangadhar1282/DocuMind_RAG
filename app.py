import os
from typing import List, Tuple, Dict

import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

from rag_utils import extract_pdf_pages, chunk_pages, retrieve_chunks

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def generate_answer(
    question: str,
    retrieved: List[Tuple[float, Dict]],
    api_key: str,
    model_name: str,
) -> str:
    """Generate an answer using only the retrieved document context."""

    if not retrieved:
        return (
            "I could not find relevant text in the uploaded document. "
            "Try a more specific question."
        )

    context = "\n\n".join(
        f"[Source: page {chunk['page']}]\n{chunk['text']}"
        for _, chunk in retrieved
    )

    client = Groq(api_key=api_key)

    prompt = f"""You are a careful document question-answering assistant.
Answer the user's question using ONLY the context below.
If the context does not contain the answer, say:
"I could not find that information in the uploaded document."
Do not invent facts. Keep the answer clear and concise.
Mention relevant source page numbers.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "Use only supplied context. Be transparent "
                    "when information is missing."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )

    return (
        response.choices[0].message.content
        or "No answer was returned."
    )


st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="wide",
)

st.title("📄 DocuMind AI")
st.caption(
    "Ask questions about a PDF. Answers are grounded in "
    "retrieved passages and show source pages."
)

with st.sidebar:
    st.header("Settings")

    api_key = st.text_input(
        "Groq API key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help=(
            "Get a key from https://console.groq.com/. "
            "Do not commit API keys to GitHub."
        ),
    )

    model_name = st.text_input(
        "Groq model",
        value="llama-3.3-70b-versatile",
    )

    top_k = st.slider(
        "Retrieved passages",
        min_value=1,
        max_value=6,
        value=4,
    )

    st.markdown("[Groq Console](https://console.groq.com/)")


uploaded_pdf = st.file_uploader(
    "Upload a text-based PDF",
    type=["pdf"],
)

if uploaded_pdf:
    if not api_key:
        st.warning(
            "Enter your Groq API key in the sidebar to generate "
            "answers. PDF retrieval can still be prepared."
        )

    try:
        with st.spinner(
            "Reading PDF and preparing semantic search…"
        ):
            pages = extract_pdf_pages(uploaded_pdf)

            if not pages:
                st.error(
                    "No extractable text found. If this is a scanned "
                    "PDF, OCR is needed first."
                )
                st.stop()

            chunks = chunk_pages(pages)
            embedding_model = SentenceTransformer(EMBEDDING_MODEL)

            st.session_state["documind_pages"] = pages
            st.session_state["documind_chunks"] = chunks
            st.session_state["documind_model"] = embedding_model
            st.session_state["documind_filename"] = uploaded_pdf.name

        st.success(
            f"Ready: {uploaded_pdf.name} · "
            f"{len(pages)} text pages · {len(chunks)} chunks"
        )

    except Exception as exc:
        st.error(f"Could not process this PDF: {exc}")


if "documind_chunks" in st.session_state:
    st.subheader(
        f"Ask about: "
        f"{st.session_state.get('documind_filename', 'your document')}"
    )

    question = st.text_input(
        "Your question",
        placeholder="Example: What are the main findings?",
    )

    if st.button(
        "Find answer",
        type="primary",
        disabled=not question.strip(),
    ):
        if not api_key:
            st.error("Add your Groq API key in the sidebar first.")

        else:
            try:
                with st.spinner(
                    "Retrieving relevant passages and generating an answer…"
                ):
                    retrieved = retrieve_chunks(
                        question,
                        st.session_state["documind_chunks"],
                        st.session_state["documind_model"],
                        top_k=top_k,
                    )

                    answer = generate_answer(
                        question,
                        retrieved,
                        api_key,
                        model_name,
                    )

                st.markdown("### Answer")
                st.write(answer)

                st.markdown("### Retrieved sources")

                for rank, (score, chunk) in enumerate(
                    retrieved,
                    start=1,
                ):
                    with st.expander(
                        f"Source {rank} · PDF page {chunk['page']} "
                        f"· similarity {score:.3f}"
                    ):
                        st.write(chunk["text"])

            except Exception as exc:
                st.error(f"Request failed: {exc}")
                st.info(
                    "Check your API key, model name, internet "
                    "connection, and provider limits."
                )


st.markdown("---")
st.caption(
    "Prototype for learning. Do not upload confidential documents "
    "or treat generated answers as guaranteed correct."
)