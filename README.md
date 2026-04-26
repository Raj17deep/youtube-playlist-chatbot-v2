# 🎬 YouTube Playlist AI Chatbot

An AI-powered chatbot that lets you **chat with any YouTube playlist or set of videos** — powered by Gemini 1.5 Flash, LangChain, LlamaIndex, and ChromaDB.

---

## ✨ Features

| Feature | Details |
|---------|---------|
| 📥 **Flexible input** | Playlist URL · single video URL · comma-separated video URLs |
| 📊 **Meta-insights dashboard** | Total views, top videos, thumbnails, durations, AI editorial summary |
| 💬 **AI chat interface** | Ask natural-language questions about your videos |
| 🔍 **Semantic transcript search** | LlamaIndex RAG pipeline over ChromaDB in-memory vector store |
| 🤖 **ReAct agent** | LangChain agent with 3 tools: `get_playlist_metadata`, `search_video_transcripts`, `summarize_single_video` |
| 🧠 **Transparent reasoning** | Expandable agent thought-process for every answer |
| 🌊 **Streaming responses** | Word-by-word token streaming to the chat interface |

---

## 🏗️ Stack

| Layer | Technology |
|-------|-----------|
| Frontend / Backend | [Streamlit](https://streamlit.io) |
| LLM | Gemini 1.5 Flash (`gemini-1.5-flash`) |
| Embeddings | Google `text-embedding-004` |
| Agentic orchestration | LangChain ReAct (`langchain`, `langchain-google-genai`) |
| RAG pipeline | LlamaIndex + ChromaDB ephemeral (in-memory) |
| Transcript scraping | `youtube-transcript-api` |
| YouTube metadata | `google-api-python-client` (YouTube Data API v3) |

---

## 🚀 Quick Start

### 1. Clone & install dependencies

```bash
git clone https://github.com/Raj17deep/youtube-playlist-chatbot-v2.git
cd youtube-playlist-chatbot-v2
pip install -r requirements.txt
```

### 2. Configure API keys

Create `.streamlit/secrets.toml` (never commit this file):

```toml
YOUTUBE_API_KEY = "your-youtube-data-api-v3-key"
GOOGLE_API_KEY  = "your-google-ai-studio-key"
```

**Getting API keys:**
- **YouTube Data API v3** → [Google Cloud Console](https://console.cloud.google.com/apis/library/youtube.googleapis.com) → Enable API → Create credentials (API key)
- **Google AI (Gemini + Embeddings)** → [Google AI Studio](https://aistudio.google.com/app/apikey) → Create API key

### 3. Run locally

```bash
streamlit run app.py
```

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push this repository to GitHub (ensure `.streamlit/secrets.toml` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo
3. In **Advanced settings → Secrets**, add:
   ```toml
   YOUTUBE_API_KEY = "your-key"
   GOOGLE_API_KEY  = "your-key"
   ```
4. Click **Deploy** — no Docker or server config required

---

## 🤖 Agent Tools

The LangChain ReAct agent has three tools:

| Tool | Description |
|------|-------------|
| `get_playlist_metadata` | Returns titles, channels, view counts, durations and transcript availability for all videos |
| `search_video_transcripts` | Semantic RAG search over ChromaDB-indexed transcripts using `text-embedding-004` |
| `summarize_single_video` | Generates a comprehensive LLM summary for one specific video (by title or ID) |

---

## ⚠️ Constraints

- ChromaDB is **ephemeral/in-memory** — data is not persisted between sessions
- Transcripts are scraped; videos with disabled captions are flagged and skipped
- All API keys are loaded via `st.secrets` only
- Gemini free tier only (no paid quotas assumed)

---

## 📁 Project Structure

```
app.py                  ← Main Streamlit application
requirements.txt        ← Python dependencies
.streamlit/
  config.toml           ← Streamlit UI configuration
  secrets.toml          ← API keys (not committed — create locally)
README.md
```