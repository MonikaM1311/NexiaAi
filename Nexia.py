# study_mate_pro_mistral_google.py
import os
import streamlit as st
import tempfile
import fitz  # PyMuPDF
import faiss
import pickle
import datetime
import requests
from sentence_transformers import SentenceTransformer
from mistralai import Mistral

# --- Streamlit page config ---
st.set_page_config(page_title="StudyMate Pro", layout="wide", page_icon="📚")

# --- Mistral API setup ---
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    st.error("Set your MISTRAL_API_KEY as an environment variable.")
    st.stop()

mistral_client = Mistral(api_key=MISTRAL_API_KEY)
MISTRAL_MODEL = "mistral-small-latest"

# --- Google CSE setup ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")
USE_GOOGLE = GOOGLE_API_KEY and GOOGLE_CSE_ID

# --- Embedding model ---
EMB_MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_FILE = "faiss.index"
STORE_FILE = "faiss_store.pkl"

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer(EMB_MODEL_NAME)

embedding_model = get_embedding_model()

# --- PDF text extraction ---
def extract_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text_content = "\n".join([page.get_text() for page in doc])
        chunks = text_content.split("\n\n")
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return []

# --- FAISS Index ---
def build_index(text_chunks):
    if not text_chunks:
        return None
    embeddings = embedding_model.encode(text_chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(index, f)
    with open(STORE_FILE, "wb") as f:
        pickle.dump(text_chunks, f)
    return index

def load_index():
    try:
        with open(INDEX_FILE, "rb") as f:
            idx = pickle.load(f)
        with open(STORE_FILE, "rb") as f:
            texts = pickle.load(f)
        return idx, texts
    except:
        return None, None

def search_index(query, index, texts, top_k=3):
    if index is None:
        return []
    q_emb = embedding_model.encode([query], convert_to_numpy=True)
    distances, ids = index.search(q_emb, top_k)
    return [texts[i] for i in ids[0]]

# --- Mistral Q&A ---
def get_mistral_answer(context, question):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    try:
        response = mistral_client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Mistral API error: {e}")
        return "Failed to get answer from Mistral."

# --- Google Search ---
def google_search(query, num_results=3):
    if not USE_GOOGLE:
        return []
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": num_results
    }
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        items = data.get("items", [])
        snippets = [item.get("snippet", "") for item in items]
        return " ".join(snippets)
    except Exception as e:
        st.warning(f"Google Search failed: {e}")
        return ""

# --- Streamlit UI ---
st.title("📚 StudyMate Pro: AI Academic Assistant")
st.markdown("Organize tasks, track streaks, summarize and ask questions from PDFs.")

# Sidebar Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Task Management", "Streak Tracker", "PDF Q&A"])

# --- Home ---
if page == "Home":
    st.write("Welcome to StudyMate Pro! Navigate using the sidebar.")
    st.markdown("""
    - **Task Management:** Add and track your daily study tasks.
    - **Streak Tracker:** See your consistency over time.
    - **PDF Q&A:** Upload PDFs, summarize content, and ask questions.
    """)

# --- Task Management ---
elif page == "Task Management":
    st.header("📝 Task Management")
    if 'tasks' not in st.session_state: st.session_state.tasks = []
    if 'completed_tasks' not in st.session_state: st.session_state.completed_tasks = []

    task_name = st.text_input("Enter a new task")
    if st.button("Add Task"):
        if task_name:
            st.session_state.tasks.append({'task': task_name, 'date': datetime.date.today(), 'done': False})
            st.success(f"Task '{task_name}' added!")
        else:
            st.error("Please enter a task name.")

    st.subheader("Your Tasks")
    if st.session_state.tasks:
        for idx, task in enumerate(st.session_state.tasks):
            col1, col2 = st.columns([0.8, 0.2])
            with col1: st.write(f"- {task['task']} (Added: {task['date']})")
            with col2:
                if st.checkbox("Done", key=idx, value=task['done']):
                    task['done'] = True
                    if task not in st.session_state.completed_tasks:
                        st.session_state.completed_tasks.append(task)
    else:
        st.info("No tasks added yet.")

# --- Streak Tracker ---
elif page == "Streak Tracker":
    st.header("🔥 Study Streak Tracker")
    if 'last_streak_date' not in st.session_state: st.session_state.last_streak_date = None
    if 'current_streak' not in st.session_state: st.session_state.current_streak = 0

    today = datetime.date.today()
    completed_today = [t for t in st.session_state.completed_tasks if t['date'] == today]
    st.metric("Tasks Completed Today", len(completed_today))

    if completed_today:
        if st.session_state.last_streak_date != today:
            if st.session_state.last_streak_date == today - datetime.timedelta(days=1) or st.session_state.last_streak_date is None:
                st.session_state.current_streak += 1
            else:
                st.session_state.current_streak = 1
            st.session_state.last_streak_date = today

    st.write(f"Current Study Streak: *{st.session_state.current_streak}* days")
    st.info("Complete at least one task per day to maintain your streak!")

# --- PDF Q&A ---
elif page == "PDF Q&A":
    st.header("📄 PDF Q&A & Summarization")
    uploaded = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            path = tmp.name

        with st.spinner("Processing PDF and building index..."):
            text_chunks = extract_text(path)
            idx = build_index(text_chunks)
            st.success("PDF processed and searchable!")

        if st.button("Generate Summary"):
            with st.spinner("Generating summary via Mistral..."):
                summary = get_mistral_answer(" ".join(text_chunks), "Summarize the PDF content")
                st.subheader("PDF Summary")
                st.write(summary)

        question = st.text_input("Ask any Queries🤔⁉️:")
        if question:
            with st.spinner("Searching PDF and Google for answer..."):
                relevant_chunks = search_index(question, idx, text_chunks)
                context_text = " ".join(relevant_chunks)
                
                # If PDF has relevant content, use it
                if context_text.strip():
                    answer = get_mistral_answer(context_text, question)
                # Else fallback to Google search
                else:
                    google_snippets = google_search(question)
                    if google_snippets:
                        answer = get_mistral_answer(google_snippets, question)
                    else:
                        answer = "No relevant content found in PDF or Google."
                
                st.subheader("Answer")
                st.write(answer)
