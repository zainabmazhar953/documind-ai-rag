
import streamlit as st
import re
import hashlib
import numpy as np
from io import BytesIO
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide"
)


st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #080D16;
    color: #EAF6FF;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #101B2B;
    border-right: 1px solid #38BDF8;
}

[data-testid="stSidebar"] * {
    color: #E0F2FE !important;
}

/* Main headings */
h1, h2, h3 {
    color: #38BDF8 !important;
    font-weight: 700 !important;
}

/* Normal text */
p, label, .stMarkdown {
    color: #DCEBFA;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background-color: #111D2E;
    border: 1px solid #23445F;
    border-radius: 12px;
    padding: 15px;
}

/* Chat input */
[data-testid="stChatInput"] {
    background-color: #101B2B;
    border: 1px solid #38BDF8;
    border-radius: 12px;
}

[data-testid="stChatInput"] textarea {
    color: #080D16 !important;
    background-color: #FFFFFF !important;
}

/* Text input */
.stTextInput input {
    background-color: #FFFFFF !important;
    color: #080D16 !important;
}

/* Buttons */
.stButton button {
    background-color: #38BDF8;
    color: #080D16 !important;
    border: none;
    border-radius: 8px;
    font-weight: bold;
}

.stButton button:hover {
    background-color: #7DD3FC;
    color: #080D16 !important;
    border: 1px solid #FFFFFF;
}

/* Upload section */
[data-testid="stFileUploader"] {
    background-color: #17263A;
    border: 1px solid #38BDF8;
    border-radius: 10px;
    padding: 10px;
}

/* Info boxes */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Slider */
.stSlider [data-baseweb="slider"] div {
    color: #38BDF8;
}

/* Divider */
hr {
    border-color: #38BDF8;
}

/* Captions */
.stCaption {
    color: #A5DDF8 !important;
}

</style>
""", unsafe_allow_html=True)


# ---------------- MODEL ----------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- SESSION ----------------

defaults = {
    "chunks": [],
    "embeddings": None,
    "messages": [],
    "fingerprint": "",
    "doc_name": ""
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---------------- PDF PROCESSING ----------------

def process_pdf(data):
    reader = PdfReader(BytesIO(data))
    chunks = []

    for page_no, page in enumerate(reader.pages, 1):
        content = page.extract_text() or ""
        content = re.sub(r"\s+", " ", content).strip()

        words = content.split()

        chunk_size = 220
        overlap = 50

        for i in range(0, len(words), chunk_size - overlap):
            part = words[i:i + chunk_size]

            if len(part) >= 10:
                chunks.append({
                    "text": " ".join(part),
                    "page": page_no
                })

    return chunks

# ---------------- SEMANTIC SEARCH ----------------

def create_embeddings(chunks, model):
    texts = [c["text"] for c in chunks]

    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

def retrieve(question, chunks, embeddings, model, top_k):
    query_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    semantic_scores = np.dot(embeddings, query_vector)

    # Combine semantic relevance with keyword matching.
    question_words = set(
        re.findall(r"\b[a-zA-Z]{3,}\b", question.lower())
    )

    final_scores = []

    for i, chunk in enumerate(chunks):
        text_words = set(
            re.findall(r"\b[a-zA-Z]{3,}\b",
                       chunk["text"].lower())
        )

        overlap = len(question_words & text_words)
        keyword_score = overlap / max(len(question_words), 1)

        score = (
            0.85 * float(semantic_scores[i])
            + 0.15 * keyword_score
        )

        final_scores.append(score)

    ranked = np.argsort(final_scores)[::-1]

    results = []

    for idx in ranked[:top_k]:
        results.append({
            "text": chunks[idx]["text"],
            "page": chunks[idx]["page"],
            "score": final_scores[idx]
        })

    return results

# ---------------- ANSWER EXTRACTION ----------------

def generate_answer(question, results, model):
    if not results:
        return "I could not find information in the uploaded PDF."

    # Extract individual sentences from retrieved passages.
    sentences = []

    for result in results:
        parts = re.split(
            r'(?<=[.!?])\s+',
            result["text"]
        )

        for sentence in parts:
            sentence = sentence.strip()

            if len(sentence.split()) >= 5:
                sentences.append({
                    "text": sentence,
                    "page": result["page"],
                    "chunk_score": result["score"]
                })

    if not sentences:
        return "Relevant passages were found, but readable sentences could not be extracted."

    # Rank sentences by semantic similarity to the question.
    sentence_texts = [s["text"] for s in sentences]

    sentence_vectors = model.encode(
        sentence_texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    scores = np.dot(sentence_vectors, query_vector)

    # Combine sentence similarity with parent chunk relevance.
    combined = []

    for i, sentence in enumerate(sentences):
        score = (
            0.80 * float(scores[i])
            + 0.20 * sentence["chunk_score"]
        )

        combined.append((score, sentence))

    combined.sort(key=lambda x: x[0], reverse=True)

    # Remove duplicate sentences.
    selected = []
    seen = set()

    for score, sentence in combined:
        normalized = sentence["text"].lower()

        if normalized not in seen:
            selected.append((score, sentence))
            seen.add(normalized)

        if len(selected) >= 5:
            break

    answer = "### Answer from your PDF\n\n"

    for score, sentence in selected:
        answer += (
            f"- {sentence['text']} "
            f"*(Page {sentence['page']})*\n\n"
        )

    answer += (
        "\n**Note:** This is an extractive answer. "
        "The information comes directly from your document."
    )

    return answer

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("⚙️ Settings")

    st.success("FREE SEMANTIC RAG MODE")

    uploaded = st.file_uploader(
        "Upload your PDF",
        type=["pdf"]
    )

    top_k = st.slider(
        "Relevant chunks",
        1, 8, 4
    )

    st.divider()

    st.markdown("### Pipeline")
    st.caption(
        "PDF → Extraction → Chunking → "
        "Embeddings → Semantic Search → Answer"
    )

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------- MAIN APPLICATION ----------------

st.title("📚 DocuMind AI")

st.write(
    "Ask questions about your documents using "
    "semantic Retrieval-Augmented Generation."
)

if uploaded:
    data = uploaded.getvalue()
    fingerprint = hashlib.md5(data).hexdigest()

    if fingerprint != st.session_state.fingerprint:
        try:
            with st.spinner("Extracting PDF content..."):
                chunks = process_pdf(data)

            if not chunks:
                st.error(
                    "No readable text found. "
                    "This PDF may be scanned or image-based."
                )
            else:
                st.session_state.chunks = chunks
                st.session_state.fingerprint = fingerprint
                st.session_state.doc_name = uploaded.name
                st.session_state.messages = []
                st.session_state.embeddings = None

        except Exception as e:
            st.error(f"PDF processing error: {e}")

if st.session_state.chunks:

    st.info(
        f"Document: {st.session_state.doc_name} "
        f"| Total chunks: {len(st.session_state.chunks)}"
    )

    try:
        with st.spinner("Loading semantic search model..."):
            model = load_model()

        if st.session_state.embeddings is None:
            with st.spinner("Creating document embeddings..."):
                st.session_state.embeddings = create_embeddings(
                    st.session_state.chunks,
                    model
                )

        st.success("Document indexed successfully!")

    except Exception as e:
        st.error(f"Model loading error: {e}")
        st.stop()

else:
    model = None
    st.warning("Upload a PDF to begin.")

# ---------------- CHAT HISTORY ----------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------- CHAT INPUT ----------------

question = st.chat_input(
    "Ask anything about your uploaded PDF..."
)

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):

        if not st.session_state.chunks:
            answer = "Please upload a PDF first."

        else:
            try:
                with st.spinner("Searching your document..."):

                    results = retrieve(
                        question,
                        st.session_state.chunks,
                        st.session_state.embeddings,
                        model,
                        top_k
                    )

                    answer = generate_answer(
                        question,
                        results,
                        model
                    )

                st.markdown(answer)

                with st.expander("View retrieved passages"):
                    for result in results:
                        st.write(
                            f"Page {result['page']} "
                            f"| Relevance: {result['score']:.3f}"
                        )
                        st.write(result["text"])
                        st.divider()

            except Exception as e:
                answer = f"Error: {e}"
                st.error(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
