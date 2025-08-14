# NEXIA.py
import os
import streamlit as st
import tempfile
import fitz  # PyMuPDF
import faiss
import pickle
import datetime
import requests
import pandas as pd
import matplotlib.pyplot as plt
import speech_recognition as sr
from sentence_transformers import SentenceTransformer
from mistralai import Mistral
from streamlit_lottie import st_lottie
from moviepy.editor import TextClip, CompositeVideoClip

# --- Streamlit page config ---
st.set_page_config(page_title="Nexia AI", layout="wide", page_icon="📚")

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

# --- Lottie helper ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

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

# --- Generate MCQs ---
def generate_mcqs(pdf_text, num_mcqs=5):
    prompt = f"Generate {num_mcqs} multiple-choice questions (with 4 options each) from the following text:\n\n{pdf_text}\n\nMCQs:"
    try:
        response = mistral_client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        mcq_text = response.choices[0].message.content
        mcqs = [line.strip("- ").strip() for line in mcq_text.split("\n") if line.strip()]
        return mcqs[:num_mcqs]
    except Exception as e:
        st.error(f"Failed to generate MCQs: {e}")
        return []

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
st.markdown("""
    <h1 style='text-align:center; color:purple;'>📚 NEXIA AI : AI Academic Assistant</h1>
    <p style='text-align:center; color:violet;'>Manage tasks, track streaks, explore PDFs, generate MCQs, and create animations.</p>
""", unsafe_allow_html=True)
st.image("https://manage.wix.com/edbe70c7-d3f2-4620-b863-8570bfc931f9", width=200)


# Sidebar Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", [
    "Home",
    "Task Management",
    "Streak Tracker",
    "PDF Summary & Search",
    "MCQ Generator",
    "Animation Generator"
])

# Initialize session state
if 'tasks' not in st.session_state: st.session_state.tasks = []
if 'completed_tasks' not in st.session_state: st.session_state.completed_tasks = []
if 'last_streak_date' not in st.session_state: st.session_state.last_streak_date = None
if 'current_streak' not in st.session_state: st.session_state.current_streak = 0

# --- Home ---
if page == "Home":
    st.write("Welcome to NEXIA AI! Navigate using the sidebar.")
    st.markdown("""
    - Task Management: Add and track your daily study tasks.
    - Streak Tracker: Visualize your study consistency.
    - PDF Summary & Search: Upload PDFs, summarize content, and ask questions.
    - MCQ Generator: Create multiple-choice questions from PDFs.
    - Animation Generator: Generate animations from a simple prompt.
    """)
    lottie_home = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_touohxv0.json")
    if lottie_home:
        st_lottie(lottie_home, height=300, key="home_lottie")

# --- Task Management ---
elif page == "Task Management":
    st.header("📝 Task Management with Priority")
    lottie_tasks = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_j1adxtyb.json")
    if lottie_tasks:
        st_lottie(lottie_tasks, height=200, key="tasks")

    task_name = st.text_input("Enter a new task")
    priority = st.selectbox("Select priority", ["Normal", "Important", "Urgent"])
    if st.button("Add Task"):
        if task_name:
            st.session_state.tasks.append({
                'task': task_name,
                'priority': priority,
                'date': datetime.date.today(),
                'done': False
            })
            st.success(f"Task '{task_name}' added!")
        else:
            st.error("Please enter a task name.")

    st.subheader("Your Tasks")
    for idx, task in enumerate(st.session_state.tasks):
        col1, col2 = st.columns([0.7, 0.3])
        with col1: st.write(f"- {task['task']} (Priority: {task['priority']}, Added: {task['date']})")
        with col2:
            if st.checkbox("Done", key=idx, value=task['done']):
                task['done'] = True
                if task not in st.session_state.completed_tasks:
                    st.session_state.completed_tasks.append(task)

    if st.button("Export Tasks & Streaks to Excel"):
        df_tasks = pd.DataFrame(st.session_state.tasks)
        df_completed = pd.DataFrame(st.session_state.completed_tasks)
        with pd.ExcelWriter("tasks_streaks.xlsx") as writer:
            df_tasks.to_excel(writer, sheet_name="All Tasks", index=False)
            df_completed.to_excel(writer, sheet_name="Completed Tasks", index=False)
        st.success("Exported tasks_streaks.xlsx!")

# --- Streak Tracker ---
elif page == "Streak Tracker":
    st.header("🔥 Study Streak Tracker")
    lottie_streak = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_u4yrau.json")
    if lottie_streak: st_lottie(lottie_streak, height=200, key="streak")

    today = datetime.date.today()
    completed_today = [t for t in st.session_state.completed_tasks if t['date'] == today]
    st.metric("Tasks Completed Today", len(completed_today))
    if completed_today:
        if st.session_state.last_streak_date != today:
            if st.session_state.last_streak_date == today - datetime.timedelta(days=1) or st.session_state.last_streak_date is None:
                st.session_state.current_streak += 1
            else: st.session_state.current_streak = 1
            st.session_state.last_streak_date = today
    st.write(f"Current Study Streak: {st.session_state.current_streak} days")
    st.info("Complete at least one task daily to maintain streak!")

    last_7_days = [(today - datetime.timedelta(days=i)) for i in range(6, -1, -1)]
    streak_counts = [len([t for t in st.session_state.completed_tasks if t['date'] == day]) for day in last_7_days]
    fig, ax = plt.subplots()
    ax.bar([day.strftime("%a") for day in last_7_days], streak_counts, color='purple')
    ax.set_title("Tasks Completed in Last 7 Days")
    ax.set_ylabel("Number of Tasks")
    st.pyplot(fig)

# --- PDF Summary & Search ---
elif page == "PDF Summary & Search":
    st.header("📄 PDF Summary & Search", divider="violet")
    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            path = tmp.name
        text_chunks = extract_text(path)
        idx = build_index(text_chunks)
        st.success("PDF processed and searchable!")

        # PDF Summary
        if st.button("Generate Summary"):
            summary = get_mistral_answer(" ".join(text_chunks), "Summarize the PDF content")
            st.subheader("PDF Summary")
            st.write(summary)
            st.download_button("Download Summary", summary, "pdf_summary.txt", "text/plain")

        # PDF Search
        query = st.text_input("Search PDF content:")
        if query:
            relevant_chunks = search_index(query, idx, text_chunks)
            context_text = " ".join(relevant_chunks)
            answer = get_mistral_answer(context_text, query) if context_text else "No relevant content found."
            st.subheader("Answer")
            st.write(answer)


# --- MCQ Generator ---
elif page == "MCQ Generator":
    st.header("📝 Generate MCQs from PDF", divider="green")
    uploaded = st.file_uploader("Upload a PDF", type="pdf", key="mcq_pdf")
    
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            path = tmp.name
        
        text_chunks = extract_text(path)
        pdf_text = " ".join(text_chunks)
        
        num_mcqs = st.number_input("Number of MCQs to generate", min_value=1, max_value=20, value=5)
        
        if st.button("Generate MCQs"):
            if pdf_text.strip():
                # Prompt Mistral to output MCQs clearly
                prompt = f"""
Generate {num_mcqs} multiple-choice questions from the following text.
Format each question as:
Q: <question>
A. <option1>
B. <option2>
C. <option3>
D. <option4>
Answer: <correct option letter>
Text:
{pdf_text}
"""
                try:
                    response = mistral_client.chat.complete(
                        model=MISTRAL_MODEL,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    mcq_text = response.choices[0].message.content
                    st.subheader("Generated MCQs")
                    # Display line by line
                    for line in mcq_text.split("\n"):
                        if line.strip():
                            st.write(line)
                except Exception as e:
                    st.error(f"Failed to generate MCQs: {e}")
            else:
                st.error("PDF is empty or could not extract text.")



# --- Animation Generator using MoviePy ---
elif page == "Animation Generator":
    st.header("🎬 Animation Generator (MoviePy)", divider="blue")
    anim_prompt = st.text_input("Enter animation prompt (e.g., 'Hello World moving text')")

    if st.button("Generate Animation"):
        if anim_prompt.strip():
            st.info(f"Generating animation for prompt: '{anim_prompt}'")
            try:
                # Create a simple text animation
                txt_clip = TextClip(anim_prompt, fontsize=70, color='purple', size=(640, 360))
                txt_clip = txt_clip.set_duration(5).set_pos(lambda t: ('center', 50 + t*20))
                
                video = CompositeVideoClip([txt_clip])
                
                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                video.write_videofile(temp_file.name, fps=24, codec='libx264', audio=False)
                
                # Display in Streamlit
                st.video(temp_file.name)
                st.success("Animation generated successfully!")
            except Exception as e:
                st.error(f"Failed to generate animation: {e}")
        else:
            st.warning("Please enter an animation prompt!")
