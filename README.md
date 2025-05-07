# [🧠 Thoughtful AI - Mini RAG Support Agent](https://thoughtfulautomation.notion.site/AI-Technical-Screen-d4d381a8c38d40fc9287cdb6c4f9992a)

This is a simple AI support assistant built using LangChain, OpenAI, FAISS, and Streamlit.  
It answers predefined questions about Thoughtful AI using a minimal Retrieval-Augmented Generation (RAG) approach.  
If no relevant document is found, it **always replies with a fixed fallback message** — as required in the original prompt.

---

## 📦 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create a `.env` file with your OpenAI key

```
OPENAI_API_KEY=your-openai-key-here
```

### 3. Run the app

```bash
streamlit run mini_rag.py
```

---

## 🧪 Try asking:

- "What does CAM do?"
- "How does PHIL work?"
- Or anything random to trigger the fallback response.

---

## ✅ Notes

- This uses FAISS + OpenAI embeddings for local similarity search.
- If the top match is too far (`distance > 0.5`), the system does **not** call the LLM and returns a fixed fallback message.
- No chunking or external database used — just a minimal and clean RAG setup.
