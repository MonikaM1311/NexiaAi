# NexiaAi
Nexia is an AI academic assistant that lets students interact with textbooks, lecture notes, and research papers conversationally. Upload PDFs and ask questions in natural language—Nexia delivers clear, contextual answers with references, making studying faster, easier, and more efficient.
# Nexia Ai 📚

**Nexia Ai** is an AI-powered academic assistant built with Streamlit. It helps you:

- Manage study tasks.
- Track daily study streaks.
- Summarize and query PDFs using AI (Mistral) and Google search.

---

## Features
# 📚 NEXIA AI - AI Academic Assistant

NEXIA AI is an **interactive academic assistant** built with Python and Streamlit. It integrates AI, PDF processing, semantic search, task management, MCQ generation, and animation creation to help students **study smarter, stay organized, and maintain streaks**.

---

## **Features**

1. **Task Management**
   - Add, track, and prioritize daily study tasks.
   - Mark tasks as done and monitor progress.
   - Export tasks and streaks to Excel.

2. **Streak Tracker**
   - Track study consistency with daily streaks.
   - Visualize progress over the last 7 days with bar charts.

3. **PDF Summary & Search**
   - Upload PDFs and extract text automatically.
   - Generate summaries using **Mistral AI**.
   - Search PDF content using **semantic search** powered by FAISS embeddings.

4. **MCQ Generator**
   - Generate multiple-choice questions from PDFs using AI.
   - Customizable number of questions and options.

5. **Animation Generator**
   - Create text animations with MoviePy directly in the app.
   - Customize prompts to generate creative educational videos.

6. **Google Search Integration (Optional)**
   - Fetch additional context using Google Custom Search API if required.

7. **Interactive UI**
   - Built with Streamlit and enhanced with **Lottie animations** for a visually appealing experience.

---

## **Installation**

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nexia-ai.git
cd nexia-ai
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
streamlit run NexiaAi.py
```
avigate through the sidebar:

Home – Overview of app features.

Task Management – Add and manage your study tasks.

Streak Tracker – Track your daily study streak.

PDF Summary & Search – Upload PDFs, summarize, and search content.

MCQ Generator – Generate multiple-choice questions from PDFs.

Animation Generator – Create simple text-based animations.

Dependencies

Python 3.10+

Streamlit

PyMuPDF (fitz)

FAISS

Pickle

Pandas

Matplotlib

SpeechRecognition

SentenceTransformers

Mistral AI SDK

Streamlit Lottie

MoviePy

Requests

Project Structure
NEXIA.py           # Main Streamlit app
faiss.index        # FAISS vector store
faiss_store.pkl    # PDF text storage
requirements.txt   # Python dependencies
README.md          # Project documentation


Home Page with interactive Lottie animation

Add tasks and track progress

Track weekly study streaks

Future Enhancements

Add voice commands for task management.

Integrate real-time quiz functionality.

Advanced animation generation (shapes, images, audio).

Dashboard analytics for PDF reading and MCQ performance.

##License

MIT License © 2025 Monika M.

##Contact

For questions or contributions, contact Monika M at:

Email: monikashivan1311@gmail.com

GitHub: github.com/MonikaM1311

“Study smarter, not harder, with NEXIA AI!”


---
