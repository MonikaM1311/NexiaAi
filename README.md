# NexiaAi
Nexia is an AI academic assistant that lets students interact with textbooks, lecture notes, and research papers conversationally. Upload PDFs and ask questions in natural language—Nexia delivers clear, contextual answers with references, making studying faster, easier, and more efficient.
# Nexia Ai 📚

**Nexia Ai** is an AI-powered academic assistant built with Streamlit. It helps you:

- Manage study tasks.
- Track daily study streaks.
- Summarize and query PDFs using AI (Mistral) and Google search.

---

## Features

### 1. Task Management
- Add, mark complete, and track study tasks.
- See completed tasks and ongoing tasks in an organized view.

### 2. Streak Tracker
- Track your daily study streak.
- Complete at least one task per day to maintain the streak.

### 3. PDF Q&A & Summarization
- Upload PDFs.
- AI summarizes content.
- Ask questions, and answers are retrieved from PDFs or Google search if content is not found locally.

---

## Setup

1. Clone the repository:

```bash
git clone <repo_url>
cd Nexia Ai
```
2.  Create a virtual environment and activate it:
```
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

3.  Install dependencies:
```
pip install -r requirements.txt
```

4.  Set your environment variables:
```
# Mistral API key
export MISTRAL_API_KEY="your_mistral_api_key"

# Google Custom Search (optional)
export GOOGLE_API_KEY="your_google_api_key"
export GOOGLE_CSE_ID="your_custom_search_engine_id"


On Windows PowerShell, use setx VARIABLE_NAME "value".
```

###Usage
```
streamlit run study_mate_pro_mistral_google.py
```

Navigate the sidebar to access Task Management, Streak Tracker, or PDF Q&A.

Upload PDFs, generate summaries, or ask questions.

Tasks can be added and marked as done to track streaks.

##Notes

PDF Handling: Uses PyMuPDF to extract text.

Embeddings: FAISS + Sentence Transformers for fast semantic search.

AI Answers: Mistral LLM is used for summarization and Q&A.

Google Search: Optional fallback if PDF does not contain relevant content.
