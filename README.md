# DocuMind AI — RAG Document Assistant

A learning project for a Decision Science / GenAI Associate portfolio. Upload a text-based PDF, ask a question, retrieve semantically similar passages using sentence embeddings, and generate a grounded answer with a Groq-hosted LLM.

## Features
- PDF text extraction with page metadata (`pypdf`)
- Modular retrieval logic in `rag_utils.py`
- Overlapping text chunking
- Semantic retrieval with Sentence Transformers (`all-MiniLM-L6-v2`)
- LLM generation through the Groq API
- Source passage display and page references
- Prompt instruction to avoid unsupported answers
- Small unit tests for chunking and empty retrieval

## Architecture
1. Extract page text from the uploaded PDF.
2. Split page text into overlapping chunks while retaining page numbers.
3. Embed chunks and the user's question.
4. Rank chunks by cosine similarity.
5. Send the top-k passages to an LLM API with a grounded-answer prompt.
6. Display the answer and the retrieved evidence.

## Run locally
Recommended: Python 3.10 or 3.11.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Get an API key from https://console.groq.com/ and enter it in the app sidebar. Alternatively, set `GROQ_API_KEY` as an environment variable before running. Never commit API keys to GitHub.

The embedding model downloads the first time the app runs, so first startup may take a few minutes and requires internet access.

## Test
```bash
pytest -q
```

The included tests check chunk overlap, page metadata, invalid chunk settings, empty retrieval, and deterministic retrieval ordering. They do not evaluate the quality of LLM-generated answers.

## Manual evaluation plan
Use a PDF whose content you understand and create a small question set before testing.

| Test | Input | Expected behavior | Record |
|---|---|---|---|
| T1: Answer present | Ask for a fact clearly stated in the PDF | Answer matches source and gives page reference | Pass/Fail + notes |
| T2: Different wording | Ask a paraphrased question | Retrieves a relevant passage | Pass/Fail + notes |
| T3: Missing answer | Ask for information absent from the PDF | Says it cannot find the information | Pass/Fail + notes |
| T4: Multiple pages | Ask a question answered across pages | Relevant pages appear among retrieved sources | Pass/Fail + notes |
| T5: Empty/scanned PDF | Upload a PDF with no selectable text | Shows a clear message; OCR is not implemented | Pass/Fail + notes |
| T6: API error | Use an invalid key or model | Shows a useful error message | Pass/Fail + notes |

Do not report a pass rate until you have actually run these tests and recorded the results. For a stronger evaluation, prepare 15–30 questions with known answers and track retrieval relevance, answer correctness, source support, and unsupported-answer behavior.

## Current limitations
- Handles text-based PDFs; OCR for scanned PDFs is not included.
- Uses in-memory processing; the document is not persisted in a vector database.
- Re-embeds chunks on each question, so this is a simple prototype rather than an optimized production system.
- Uses an external LLM API. Do not upload confidential or sensitive documents.
- Similarity score is a retrieval score, not a probability that the answer is correct.
- Model names and API limits can change; check the provider console if a request fails.

## Resume bullet — use only after you have built and tested it
**DocuMind AI — RAG Document Assistant:** Built a Python/Streamlit application that extracts PDF text, chunks documents, performs semantic retrieval with sentence embeddings, and uses an LLM API to generate context-grounded answers with source-page references. Evaluated retrieval and answer behavior using a documented test set.

Only include the final sentence about evaluation after completing and recording the tests. Describe the exact tools and results you personally verified.
