"""
YouTube Playlist AI Chatbot
============================
Stack : Streamlit · LangChain ReAct · LlamaIndex · ChromaDB (in-memory)
        Gemini 1.5 Flash · text-embedding-004 · youtube-transcript-api
        google-api-python-client
"""

# ── page config must come first ─────────────────────────────────────────────
import streamlit as st

st.set_page_config(
    page_title="🎬 YouTube Playlist AI Chatbot",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── standard library ─────────────────────────────────────────────────────────
import os
import re
import time
import traceback
from typing import Optional
from urllib.parse import urlparse, parse_qs

# ── google youtube api ────────────────────────────────────────────────────────
from googleapiclient.discovery import build as _yt_build
from googleapiclient.errors import HttpError

# ── transcript scraping ───────────────────────────────────────────────────────
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)

# ── vector store ──────────────────────────────────────────────────────────────
import chromadb
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.google import GeminiEmbedding

# ── langchain / agent ─────────────────────────────────────────────────────────
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  URL / ID  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_playlist_id(url: str) -> Optional[str]:
    """Return playlist ID from any YouTube playlist URL, else None."""
    params = parse_qs(urlparse(url.strip()).query)
    return params.get("list", [None])[0]


def _extract_video_id(url: str) -> Optional[str]:
    """Return 11-char video ID from any YouTube URL or bare ID, else None."""
    url = url.strip()
    # youtu.be/<id>
    m = re.match(r"(?:https?://)?youtu\.be/([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    # watch?v=<id>
    params = parse_qs(urlparse(url).query)
    if "v" in params:
        return params["v"][0]
    # /shorts/<id>
    m = re.match(
        r"(?:https?://)?(?:www\.)?youtube\.com/shorts/([A-Za-z0-9_-]{11})", url
    )
    if m:
        return m.group(1)
    # bare 11-char ID
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  DURATION  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_iso_duration(dur: str) -> str:
    """ISO 8601 duration → human-readable string (e.g. '1:23:45' or '4:30')."""
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", dur or "")
    if not m:
        return "0:00"
    h, mi, s = (int(x or 0) for x in m.groups())
    return f"{h}:{mi:02d}:{s:02d}" if h else f"{mi}:{s:02d}"


def _iso_to_seconds(dur: str) -> int:
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", dur or "")
    if not m:
        return 0
    h, mi, s = (int(x or 0) for x in m.groups())
    return h * 3600 + mi * 60 + s


def _fmt_seconds(total: int) -> str:
    h, rem = divmod(total, 3600)
    m = rem // 60
    return f"{h}h {m}m" if h else f"{m}m"


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  YOUTUBE  DATA  API
# ═══════════════════════════════════════════════════════════════════════════════

def _yt_client(api_key: str):
    return _yt_build("youtube", "v3", developerKey=api_key)


def _get_playlist_video_ids(playlist_id: str, yt) -> list[str]:
    """Fetch all video IDs from a playlist, handling pagination."""
    ids, token = [], None
    while True:
        resp = (
            yt.playlistItems()
            .list(
                part="contentDetails",
                playlistId=playlist_id,
                maxResults=50,
                pageToken=token,
            )
            .execute()
        )
        for item in resp.get("items", []):
            ids.append(item["contentDetails"]["videoId"])
        token = resp.get("nextPageToken")
        if not token:
            break
    return ids


def _fetch_video_metadata(ids: list[str], yt) -> list[dict]:
    """Fetch snippet, statistics, and contentDetails for up to N videos."""
    videos = []
    for i in range(0, len(ids), 50):
        batch = ids[i : i + 50]
        resp = (
            yt.videos()
            .list(
                part="snippet,statistics,contentDetails",
                id=",".join(batch),
            )
            .execute()
        )
        for item in resp.get("items", []):
            vid_id = item["id"]
            sn = item.get("snippet", {})
            stats = item.get("statistics", {})
            cd = item.get("contentDetails", {})
            thumbs = sn.get("thumbnails", {})
            thumb = (
                thumbs.get("high", {}).get("url")
                or thumbs.get("medium", {}).get("url")
                or thumbs.get("default", {}).get("url", "")
            )
            dur = cd.get("duration", "PT0S")
            videos.append(
                {
                    "video_id": vid_id,
                    "title": sn.get("title", "Unknown"),
                    "channel": sn.get("channelTitle", "Unknown"),
                    "description": sn.get("description", "")[:500],
                    "published_at": sn.get("publishedAt", ""),
                    "thumbnail": thumb,
                    "views": int(stats.get("viewCount", 0)),
                    "likes": int(stats.get("likeCount", 0)),
                    "duration_iso": dur,
                    "duration": _parse_iso_duration(dur),
                    "duration_seconds": _iso_to_seconds(dur),
                    "url": f"https://www.youtube.com/watch?v={vid_id}",
                }
            )
    return videos


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  TRANSCRIPT  SCRAPING
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_transcripts(ids: list[str]) -> tuple[dict[str, str], list[str]]:
    """
    Attempt to fetch transcripts for every video ID.

    Returns
    -------
    ok   : {video_id: transcript_text}
    fail : [video_id, ...]  – videos with no available transcript
    """
    ok: dict[str, str] = {}
    fail: list[str] = []
    for vid_id in ids:
        try:
            raw = YouTubeTranscriptApi.get_transcript(vid_id)
            text = " ".join(
                e["text"].strip() for e in raw if e["text"].strip()
            )
            if text:
                ok[vid_id] = text
            else:
                fail.append(vid_id)
        except (TranscriptsDisabled, NoTranscriptFound, Exception):
            fail.append(vid_id)
    return ok, fail


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  VECTOR INDEX  (LlamaIndex + ChromaDB in-memory)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_index(
    videos: list[dict],
    transcripts: dict[str, str],
    google_api_key: str,
) -> VectorStoreIndex:
    """Build an ephemeral ChromaDB vector index from transcript documents."""
    # Make API key available to the google SDK
    os.environ["GOOGLE_API_KEY"] = google_api_key

    # Configure LlamaIndex global settings
    Settings.embed_model = GeminiEmbedding(
        model_name="models/text-embedding-004",
        api_key=google_api_key,
    )
    Settings.llm = None  # LLM handled by LangChain

    # Ephemeral (in-memory, session-scoped) ChromaDB
    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.get_or_create_collection("yt_transcripts")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    vid_map = {v["video_id"]: v for v in videos}
    docs = []
    for vid_id, text in transcripts.items():
        meta = vid_map.get(vid_id, {})
        docs.append(
            Document(
                text=text,
                metadata={
                    "video_id": vid_id,
                    "title": meta.get("title", "Unknown"),
                    "channel": meta.get("channel", "Unknown"),
                    "url": meta.get("url", ""),
                    "views": str(meta.get("views", 0)),
                    "duration": meta.get("duration", ""),
                },
                doc_id=vid_id,
            )
        )

    return VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_ctx,
        show_progress=False,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  LANGCHAIN  TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

def _make_tools(
    videos: list[dict],
    transcripts: dict[str, str],
    index: Optional[VectorStoreIndex],
    llm: ChatGoogleGenerativeAI,
) -> list[Tool]:
    """Return the 3 LangChain tools for the ReAct agent (closures over state)."""

    # ── Tool 1: playlist metadata ────────────────────────────────────────────
    def get_playlist_metadata(query: str) -> str:  # noqa: ARG001
        lines = [f"This playlist/collection has **{len(videos)}** video(s):\n"]
        for v in videos:
            badge = "✅ transcript" if v["video_id"] in transcripts else "❌ no transcript"
            lines.append(
                f"• **{v['title']}** — {v['channel']}\n"
                f"  Views: {v['views']:,} | Duration: {v['duration']} | {badge}\n"
                f"  URL: {v['url']}\n"
            )
        return "".join(lines)

    # ── Tool 2: semantic transcript search (RAG) ─────────────────────────────
    def search_video_transcripts(query: str) -> str:
        if not transcripts:
            return "No transcripts are available for search."
        if index is None:
            return "Vector index is not available (no transcripts were indexed)."
        qe = index.as_query_engine(similarity_top_k=3)
        resp = qe.query(query)
        out = f"**Results for:** *{query}*\n\n{resp}\n"
        if hasattr(resp, "source_nodes") and resp.source_nodes:
            out += "\n**Sources:**\n"
            for node in resp.source_nodes:
                md = node.node.metadata
                out += f"- {md.get('title', '?')} — {md.get('url', '')}\n"
        return out

    # ── Tool 3: single-video summariser ─────────────────────────────────────
    def summarize_single_video(identifier: str) -> str:
        vid: Optional[dict] = None
        tr_text: Optional[str] = None

        # exact video_id match
        if identifier in transcripts:
            tr_text = transcripts[identifier]
            vid = next((v for v in videos if v["video_id"] == identifier), None)

        # partial title match (case-insensitive)
        if not vid:
            lo = identifier.lower()
            for v in videos:
                if lo in v["title"].lower() and v["video_id"] in transcripts:
                    vid = v
                    tr_text = transcripts[v["video_id"]]
                    break

        if not vid:
            available = [v["title"] for v in videos if v["video_id"] in transcripts]
            return (
                f"Could not find a video matching '{identifier}'. "
                f"Available titles: {', '.join(available[:5]) or 'none'}"
            )
        if not tr_text:
            return f"No transcript available for '{vid['title']}'."

        # Truncate to stay within token limits
        if len(tr_text) > 10_000:
            tr_text = tr_text[:10_000] + "…[truncated]"

        msgs = [
            SystemMessage(content="You are an expert YouTube video summariser."),
            HumanMessage(
                content=(
                    f"Write a comprehensive summary of the following video.\n\n"
                    f"**Title:** {vid['title']}\n"
                    f"**Channel:** {vid['channel']}\n\n"
                    f"**Transcript:**\n{tr_text}"
                )
            ),
        ]
        result = llm.invoke(msgs)
        return f"**Summary — {vid['title']}**\n\n{result.content}"

    return [
        Tool(
            name="get_playlist_metadata",
            func=get_playlist_metadata,
            description=(
                "Returns metadata for every video: title, channel, view count, "
                "duration, and transcript availability. "
                "Use this for overview or statistics questions."
            ),
        ),
        Tool(
            name="search_video_transcripts",
            func=search_video_transcripts,
            description=(
                "Semantically searches video transcripts to find relevant spoken "
                "content. Use this when the user asks what was discussed or "
                "mentioned in the videos."
            ),
        ),
        Tool(
            name="summarize_single_video",
            func=summarize_single_video,
            description=(
                "Generates a comprehensive summary of one specific video. "
                "Input: video title (partial match accepted) or video ID."
            ),
        ),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  LANGCHAIN  REACT  AGENT
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  LANGGRAPH  REACT  AGENT
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = (
    "You are a knowledgeable AI assistant for analysing YouTube video "
    "collections. Use the available tools to answer questions about the "
    "videos. Always use get_playlist_metadata first for overview questions, "
    "search_video_transcripts for content questions, and "
    "summarize_single_video for video-specific summaries."
)


def _build_agent(tools: list[Tool], llm: ChatGoogleGenerativeAI):
    """Build a LangGraph ReAct agent graph."""
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=_SYSTEM_PROMPT,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  PROCESSING  PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def _process_input(url_text: str, yt_key: str, g_key: str) -> None:
    """End-to-end pipeline: parse → metadata → transcripts → index → agent."""
    prog = st.progress(0)
    status = st.empty()

    try:
        # ── resolve video IDs ───────────────────────────────────────────────
        status.text("🔍 Resolving video IDs…")
        yt = _yt_client(yt_key)
        url_text = url_text.strip()
        video_ids: list[str] = []

        playlist_id = _extract_playlist_id(url_text)
        if playlist_id:
            status.text("📋 Fetching playlist items…")
            video_ids = _get_playlist_video_ids(playlist_id, yt)
        elif "," in url_text:
            for part in url_text.split(","):
                vid = _extract_video_id(part.strip())
                if vid:
                    video_ids.append(vid)
        else:
            vid = _extract_video_id(url_text)
            if vid:
                video_ids.append(vid)

        if not video_ids:
            st.error(
                "❌ No valid YouTube video IDs found. "
                "Please check the URL and try again."
            )
            return

        prog.progress(15)

        # ── fetch metadata ──────────────────────────────────────────────────
        status.text(f"📊 Fetching metadata for {len(video_ids)} video(s)…")
        videos = _fetch_video_metadata(video_ids, yt)
        if not videos:
            st.error("❌ Could not fetch video metadata. Check your YouTube API key.")
            return

        prog.progress(30)

        # ── fetch transcripts ───────────────────────────────────────────────
        status.text(f"📝 Fetching transcripts for {len(video_ids)} video(s)…")
        transcripts, failed = _fetch_transcripts(video_ids)
        prog.progress(55)

        if failed:
            failed_titles = [
                next((v["title"] for v in videos if v["video_id"] == f), f)
                for f in failed
            ]
            st.warning(
                f"⚠️ {len(failed)} video(s) have no available transcript: "
                + ", ".join(f"*{t}*" for t in failed_titles[:3])
                + ("…" if len(failed_titles) > 3 else "")
            )

        # ── build vector index ──────────────────────────────────────────────
        index: Optional[VectorStoreIndex] = None
        if transcripts:
            status.text(
                f"🔢 Building vector index for {len(transcripts)} transcript(s)…"
            )
            index = _build_index(videos, transcripts, g_key)
        else:
            st.warning("⚠️ No transcripts available – search will be unavailable.")
        prog.progress(80)

        # ── initialise LLM + agent ──────────────────────────────────────────
        status.text("🤖 Initialising Gemini 1.5 Flash agent…")
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=g_key,
            temperature=0.7,
            streaming=True,
        )
        tools = _make_tools(videos, transcripts, index, llm)
        agent_exec = _build_agent(tools, llm)

        # ── persist in session state ────────────────────────────────────────
        st.session_state.update(
            videos=videos,
            transcripts=transcripts,
            failed_transcripts=failed,
            index=index,
            llm=llm,
            agent_executor=agent_exec,
            messages=[],
            editorial_summary=None,
        )

        prog.progress(100)
        status.text("✅ Done!")
        st.success(
            f"✅ Loaded **{len(videos)}** videos "
            f"({len(transcripts)} with transcripts)."
        )
        time.sleep(0.6)
        st.rerun()

    except HttpError as exc:
        st.error(f"YouTube API error: {exc}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected error: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  DASHBOARD  PAGE
# ═══════════════════════════════════════════════════════════════════════════════

def _page_dashboard() -> None:
    vids: list[dict] = st.session_state.videos
    trs: dict[str, str] = st.session_state.transcripts
    llm = st.session_state.get("llm")

    st.header("📊 Meta-Insights Dashboard")

    # ── KPI metrics ─────────────────────────────────────────────────────────
    total_views = sum(v["views"] for v in vids)
    total_secs = sum(v["duration_seconds"] for v in vids)
    tr_count = sum(1 for v in vids if v["video_id"] in trs)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📹 Videos", len(vids))
    k2.metric("👁️ Total Views", f"{total_views:,}")
    k3.metric("⏱️ Total Duration", _fmt_seconds(total_secs))
    k4.metric("📝 Transcripts", f"{tr_count}/{len(vids)}")

    st.divider()

    # ── top 5 videos ─────────────────────────────────────────────────────────
    st.subheader("🏆 Top Videos by Views")
    top = sorted(vids, key=lambda x: x["views"], reverse=True)[:5]
    cols = st.columns(len(top))
    for col, v in zip(cols, top):
        with col:
            if v["thumbnail"]:
                st.image(v["thumbnail"], use_column_width=True)
            title = v["title"]
            st.markdown(
                f"**{title[:42]}…**" if len(title) > 42 else f"**{title}**"
            )
            st.caption(f"👁️ {v['views']:,} views")
            st.caption(f"⏱️ {v['duration']}  |  📺 {v['channel'][:20]}")

    st.divider()

    # ── full video list ───────────────────────────────────────────────────────
    st.subheader("📋 All Videos")
    for v in vids:
        has_tr = v["video_id"] in trs
        badge = "✅" if has_tr else "❌"
        with st.expander(f"{badge} {v['title']}", expanded=False):
            c1, c2 = st.columns([1, 3])
            with c1:
                if v["thumbnail"]:
                    st.image(v["thumbnail"])
            with c2:
                st.markdown(f"**Channel:** {v['channel']}")
                st.markdown(f"**Views:** {v['views']:,}")
                st.markdown(f"**Duration:** {v['duration']}")
                st.markdown(
                    f"**Transcript:** "
                    f"{'Available ✅' if has_tr else 'Not available ❌'}"
                )
                st.markdown(f"[🔗 Watch on YouTube]({v['url']})")

    st.divider()

    # ── AI editorial summary ─────────────────────────────────────────────────
    st.subheader("🤖 AI Editorial Summary")

    if "editorial_summary" not in st.session_state:
        st.session_state.editorial_summary = None

    if st.session_state.editorial_summary:
        st.markdown(st.session_state.editorial_summary)
    elif llm:
        if st.button("✨ Generate AI Editorial Summary", key="gen_editorial"):
            with st.spinner("Generating editorial summary…"):
                video_list = "\n".join(
                    f"- {v['title']} by {v['channel']} "
                    f"({v['views']:,} views, {v['duration']})"
                    for v in vids[:20]
                )
                try:
                    msgs = [
                        SystemMessage(
                            content="You are an expert content curator and editor."
                        ),
                        HumanMessage(
                            content=(
                                "Provide a concise but comprehensive editorial "
                                "summary of this YouTube collection including:\n"
                                "1. Overall theme/topic\n"
                                "2. Key insights and takeaways\n"
                                "3. Educational value assessment\n"
                                "4. Recommended viewing order (if applicable)\n\n"
                                f"Videos:\n{video_list}"
                            )
                        ),
                    ]
                    st.session_state.editorial_summary = llm.invoke(msgs).content
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Failed to generate summary: {exc}")
    else:
        st.info("Configure API keys to generate AI editorial summaries.")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. CHAT  PAGE
# ═══════════════════════════════════════════════════════════════════════════════

def _page_chat() -> None:
    agent_graph = st.session_state.get("agent_executor")
    if not agent_graph:
        st.info("Agent is not ready yet – please process a URL first.")
        return

    st.header("💬 Chat with Your Playlist")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── render history ────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("steps"):
                with st.expander("🔍 Agent Reasoning", expanded=False):
                    st.markdown(msg["steps"])

    # ── new user input ────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask anything about your YouTube content…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            answer_box = st.empty()

            try:
                with st.spinner("Thinking…"):
                    result = agent_graph.invoke(
                        {"messages": [HumanMessage(content=prompt)]}
                    )

                # Extract final answer from the last AIMessage
                final = ""
                all_messages = result.get("messages", [])
                for m in reversed(all_messages):
                    if isinstance(m, AIMessage) and m.content:
                        final = m.content
                        break
                if not final:
                    final = "I couldn't generate a response."

                # Stream final answer word-by-word for a live-typing effect
                def _word_stream(text: str):
                    for word in text.split():
                        yield word + " "
                        time.sleep(0.025)

                answer_box.write_stream(_word_stream(final))

                # Build reasoning summary from tool-call messages
                steps_md = ""
                for m in all_messages:
                    if isinstance(m, AIMessage) and m.tool_calls:
                        for tc in m.tool_calls:
                            steps_md += (
                                f"**🔧 Tool:** `{tc['name']}`\n\n"
                                f"**Input:** {tc['args']}\n\n"
                            )
                    elif isinstance(m, ToolMessage):
                        excerpt = str(m.content)[:400]
                        steps_md += f"**Output (excerpt):** {excerpt}…\n\n---\n\n"

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": final,
                        "steps": steps_md,
                    }
                )

            except Exception as exc:  # noqa: BLE001
                err = f"⚠️ Error: {exc}"
                answer_box.error(err)
                st.session_state.messages.append(
                    {"role": "assistant", "content": err}
                )


# ═══════════════════════════════════════════════════════════════════════════════
# 11. LANDING  PAGE
# ═══════════════════════════════════════════════════════════════════════════════

def _landing() -> None:
    st.markdown(
        """
## 👋 Welcome to YouTube Playlist AI Chatbot!

Load a YouTube playlist or video(s) using the **sidebar** to get started.

---

### 📥 Supported inputs

| Type | Example |
|------|---------|
| Playlist URL | `https://youtube.com/playlist?list=PL…` |
| Single video | `https://youtube.com/watch?v=dQw4w9WgXcQ` |
| Short link | `https://youtu.be/dQw4w9WgXcQ` |
| Multiple videos | `url1, url2, url3` |

---

### ✨ Features

- 📊 **Dashboard** – total views, top videos, thumbnails, durations, AI editorial summary
- 💬 **Chat** – ask anything; the ReAct agent answers using transcript search & metadata
- 🔍 **Semantic search** – LlamaIndex + ChromaDB (in-memory) over video transcripts
- 🧠 **Transparent reasoning** – expandable agent thought process for every answer
- 🌊 **Streaming responses** – word-by-word token streaming to the chat interface

---

### 🔑 Required API keys (set in Streamlit secrets)

| Secret key | Where to get it |
|-----------|-----------------|
| `YOUTUBE_API_KEY` | [Google Cloud Console → YouTube Data API v3](https://console.cloud.google.com/apis/) |
| `GOOGLE_API_KEY`  | [Google AI Studio](https://aistudio.google.com/app/apikey) |
        """
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 12. MAIN  ENTRY  POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.title("🎬 YouTube Playlist AI Chatbot")
    st.caption(
        "Powered by **Gemini 1.5 Flash** · **LangChain ReAct** · "
        "**LlamaIndex** · **ChromaDB** · **text-embedding-004**"
    )

    # ── sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Load API keys from Streamlit secrets
        try:
            yt_key: str = st.secrets["YOUTUBE_API_KEY"]
            g_key: str = st.secrets["GOOGLE_API_KEY"]
            st.success("✅ API keys loaded")
        except (KeyError, FileNotFoundError):
            st.error("❌ API keys not found in secrets")
            st.info(
                "**Local setup** – create `.streamlit/secrets.toml`:\n"
                "```toml\n"
                'YOUTUBE_API_KEY = "your-key"\n'
                'GOOGLE_API_KEY  = "your-key"\n'
                "```\n"
                "**Streamlit Cloud** – add secrets in the app dashboard."
            )
            st.stop()

        st.divider()
        st.header("📥 Load Content")

        url_input = st.text_area(
            "YouTube URL(s)",
            placeholder=(
                "Playlist: https://youtube.com/playlist?list=…\n"
                "Video:    https://youtube.com/watch?v=…\n"
                "Multiple: url1, url2, url3"
            ),
            height=110,
            key="url_input_box",
        )

        process_clicked = st.button(
            "🚀 Process",
            type="primary",
            disabled=not bool(url_input and url_input.strip()),
        )
        if process_clicked:
            _process_input(url_input, yt_key, g_key)

        if st.session_state.get("videos"):
            st.divider()
            n = len(st.session_state.videos)
            t = len(st.session_state.transcripts)
            st.success(f"✅ {n} videos loaded  ({t} transcripts)")

            if st.button("🗑️ Clear Session"):
                for k in [
                    "videos",
                    "transcripts",
                    "failed_transcripts",
                    "index",
                    "llm",
                    "agent_executor",
                    "messages",
                    "editorial_summary",
                ]:
                    st.session_state.pop(k, None)
                st.rerun()

    # ── main content area ─────────────────────────────────────────────────────
    if not st.session_state.get("videos"):
        _landing()
        return

    tab1, tab2 = st.tabs(["📊 Dashboard", "💬 Chat"])
    with tab1:
        _page_dashboard()
    with tab2:
        _page_chat()


if __name__ == "__main__":
    main()
