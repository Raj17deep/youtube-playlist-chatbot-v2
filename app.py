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
    page_title="PlaylistIQ – Chat with Your YouTube Content",
    page_icon="▶️",
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
    def get_playlist_metadata(query: str) -> str:
        """Return metadata for all videos, optionally filtered by the query."""
        query_lower = query.lower() if query.strip() else ""
        lines = [f"This playlist/collection has **{len(videos)}** video(s):\n"]
        for v in videos:
            # Apply lightweight filter: include all videos if query is generic,
            # or include only matching ones when a specific title/channel is given.
            if query_lower and query_lower not in v["title"].lower() and query_lower not in v["channel"].lower():
                continue
            badge = "✅ transcript" if v["video_id"] in transcripts else "❌ no transcript"
            lines.append(
                f"• **{v['title']}** — {v['channel']}\n"
                f"  Views: {v['views']:,} | Duration: {v['duration']} | {badge}\n"
                f"  URL: {v['url']}\n"
            )
        # Fall back to full list if the filter matched nothing
        if len(lines) == 1:
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

    st.header("Overview")

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
    st.subheader("Top Videos by Views")
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
    st.subheader("All Videos")
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
    st.subheader("AI Editorial Summary")

    if "editorial_summary" not in st.session_state:
        st.session_state.editorial_summary = None

    if st.session_state.editorial_summary:
        st.markdown(st.session_state.editorial_summary)
    elif llm:
        if st.button("Generate Summary", key="gen_editorial"):
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
        st.info("Load a playlist or video to generate an AI editorial summary.")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. CHAT  PAGE
# ═══════════════════════════════════════════════════════════════════════════════

def _page_chat() -> None:
    agent_graph = st.session_state.get("agent_executor")
    if not agent_graph:
        st.info("Load a playlist or video first to start chatting.")
        return

    st.header("Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── render history ────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("steps"):
                with st.expander("View sources & reasoning", expanded=False):
                    st.markdown(msg["steps"])

    # ── new user input ────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask anything about your videos…"):
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
<div class="hero-block">
  <h1>Understand any YouTube playlist — instantly.</h1>
  <p class="hero-sub">Paste a playlist or video link on the left, then ask questions, get summaries, and uncover insights in seconds.</p>
</div>

<div class="feature-grid">
  <div class="feature-card">
    <div class="feature-icon">📊</div>
    <div class="feature-title">Instant Dashboard</div>
    <div class="feature-desc">See total views, top-performing videos, durations, and thumbnails at a glance.</div>
  </div>
  <div class="feature-card">
    <div class="feature-icon">💬</div>
    <div class="feature-title">Conversational Q&A</div>
    <div class="feature-desc">Ask natural-language questions about any video in the playlist and get accurate, sourced answers.</div>
  </div>
  <div class="feature-card">
    <div class="feature-icon">📝</div>
    <div class="feature-title">Video Summaries</div>
    <div class="feature-desc">Get a crisp AI-generated summary for any individual video without watching it.</div>
  </div>
  <div class="feature-card">
    <div class="feature-icon">🔍</div>
    <div class="feature-title">Deep Search</div>
    <div class="feature-desc">Search across all transcripts to find exactly where a topic was discussed.</div>
  </div>
</div>

<div class="supported-block">
  <h3>Supported link formats</h3>
  <table class="supported-table">
    <tr><td>Playlist</td><td><code>https://youtube.com/playlist?list=PL…</code></td></tr>
    <tr><td>Single video</td><td><code>https://youtube.com/watch?v=…</code></td></tr>
    <tr><td>Short link</td><td><code>https://youtu.be/…</code></td></tr>
    <tr><td>Multiple videos</td><td><code>url1, url2, url3</code></td></tr>
  </table>
</div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 12. MAIN  ENTRY  POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── inject custom CSS ─────────────────────────────────────────────────────
    # Design system: "PlaylistIQ Obsidian" — generated via Stitch MCP
    # Palette: bg #0B0D14 · surface #1A1D2E · accent #6C63FF→#8781FF · Inter
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        -webkit-font-smoothing: antialiased;
    }

    /* ── hide default Streamlit chrome ── */
    #MainMenu, footer { visibility: hidden; }
    header { visibility: hidden; }

    /* ════════════════════════════════════════════
       APP BACKGROUND — Layer 0
    ════════════════════════════════════════════ */
    .stApp {
        background: #0B0D14;
        color: #E2E2EC;
    }

    /* ════════════════════════════════════════════
       SIDEBAR — Layer 1 (Control Center)
    ════════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: #1A1D2E;
        border-right: 1px solid rgba(37, 40, 64, 0.8);
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    /* Sidebar text inputs & textarea */
    [data-testid="stSidebar"] .stTextArea textarea,
    [data-testid="stSidebar"] .stTextInput input {
        background: #11131A;
        border: 1px solid #252840;
        color: #E2E2EC;
        border-radius: 10px;
        font-size: 0.875rem;
        line-height: 1.6;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    [data-testid="stSidebar"] .stTextArea textarea:focus,
    [data-testid="stSidebar"] .stTextInput input:focus {
        border-color: rgba(108, 99, 255, 0.5);
        box-shadow: 0 0 0 3px rgba(108, 99, 255, 0.12);
        outline: none;
    }

    /* Primary CTA button — gradient pill */
    [data-testid="stSidebar"] .stButton > button[kind="primary"],
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #6C63FF 0%, #8781FF 100%);
        color: #fff;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.01em;
        width: 100%;
        padding: 0.65rem 1rem;
        box-shadow: 0 4px 20px rgba(108, 99, 255, 0.25);
        transition: box-shadow 0.25s, transform 0.15s, opacity 0.2s;
        cursor: pointer;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        box-shadow: 0 6px 28px rgba(108, 99, 255, 0.4);
        transform: translateY(-1px);
        opacity: 0.95;
    }
    [data-testid="stSidebar"] .stButton > button:active {
        transform: translateY(0px);
        box-shadow: 0 2px 10px rgba(108, 99, 255, 0.3);
    }
    [data-testid="stSidebar"] .stButton > button:disabled {
        opacity: 0.35;
        cursor: not-allowed;
        transform: none;
        box-shadow: none;
    }

    /* Secondary / ghost button (Start over) */
    [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        background: transparent;
        border: 1px solid #252840;
        color: #8B8FAA;
        box-shadow: none;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
        background: #252840;
        color: #E2E2EC;
        box-shadow: none;
        transform: none;
    }

    /* Success alert in sidebar */
    [data-testid="stSidebar"] .stAlert {
        background: rgba(34, 197, 94, 0.08) !important;
        border: 1px solid rgba(34, 197, 94, 0.25) !important;
        border-radius: 10px !important;
        color: #4ADE80 !important;
        font-size: 0.82rem;
        font-weight: 500;
    }

    /* ── sidebar logo / brand ── */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 0.2rem 0 1.4rem 0;
        border-bottom: 1px solid rgba(37, 40, 64, 0.9);
        margin-bottom: 1.4rem;
    }
    .sidebar-brand .brand-icon-wrap {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        background: linear-gradient(135deg, #6C63FF 0%, #8781FF 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        flex-shrink: 0;
        box-shadow: 0 4px 16px rgba(108, 99, 255, 0.35);
    }
    .sidebar-brand .brand-name {
        font-size: 1.05rem;
        font-weight: 700;
        color: #E2E2EC;
        letter-spacing: -0.025em;
        line-height: 1.2;
    }
    .sidebar-brand .brand-tagline {
        font-size: 0.7rem;
        color: #8B8FAA;
        margin-top: 2px;
        letter-spacing: 0.01em;
    }

    /* ── section labels in sidebar ── */
    .sidebar-section-label {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #4D5175;
        margin: 1.2rem 0 0.6rem 0;
    }

    /* ════════════════════════════════════════════
       LANDING PAGE — Hero & Features
    ════════════════════════════════════════════ */
    .hero-block {
        text-align: center;
        padding: 5rem 2rem 3rem 2rem;
    }
    .hero-block h1 {
        font-size: 2.75rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        color: #E2E2EC;
        margin-bottom: 0.75rem;
        line-height: 1.15;
    }
    .hero-sub {
        display: block;
        font-size: 1.05rem;
        font-weight: 400;
        color: #8B8FAA;
        max-width: 540px;
        margin: 0.6rem auto 0 auto;
        line-height: 1.7;
        text-align: center;
    }

    /* feature cards grid */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        max-width: 960px;
        margin: 3rem auto 0 auto;
    }
    @media (max-width: 860px) {
        .feature-grid { grid-template-columns: repeat(2, 1fr); }
        .hero-block h1 { font-size: 2rem; }
    }
    .feature-card {
        background: #1A1D2E;
        border: 1px solid #252840;
        border-radius: 16px;
        padding: 1.5rem 1.2rem;
        text-align: center;
        cursor: default;
        transition: border-color 0.25s, transform 0.2s, box-shadow 0.25s;
    }
    .feature-card:hover {
        border-color: rgba(108, 99, 255, 0.35);
        transform: translateY(-3px);
        box-shadow: 0 12px 32px rgba(108, 99, 255, 0.08);
    }
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.65rem;
        display: block;
    }
    .feature-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #E2E2EC;
        margin-bottom: 0.4rem;
        letter-spacing: -0.01em;
    }
    .feature-desc {
        font-size: 0.8rem;
        color: #8B8FAA;
        line-height: 1.55;
    }

    /* supported links card */
    .supported-block {
        max-width: 720px;
        margin: 3.5rem auto;
        padding: 1.75rem 2rem;
        background: #1A1D2E;
        border: 1px solid #252840;
        border-radius: 16px;
    }
    .supported-block h3 {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #4D5175;
        margin-bottom: 1.1rem;
    }
    .supported-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    .supported-table td {
        padding: 0.5rem 0.7rem;
        border-bottom: 1px solid rgba(37, 40, 64, 0.8);
        color: #8B8FAA;
        vertical-align: middle;
    }
    .supported-table tr:last-child td { border-bottom: none; }
    .supported-table td:first-child { color: #C7C4D8; font-weight: 500; width: 140px; }
    .supported-table code {
        background: #11131A;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.78rem;
        color: #A8B4CC;
        font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    }

    /* ════════════════════════════════════════════
       KPI METRIC CARDS — Purple left-border accent
    ════════════════════════════════════════════ */
    [data-testid="stMetric"] {
        background: #1A1D2E;
        border: 1px solid #252840;
        border-left: 3px solid #6C63FF;
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 4px 20px rgba(108, 99, 255, 0.05);
        transition: box-shadow 0.25s, transform 0.2s;
    }
    [data-testid="stMetric"]:hover {
        box-shadow: 0 8px 32px rgba(108, 99, 255, 0.1);
        transform: translateY(-2px);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.65rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: #4D5175 !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #E2E2EC !important;
        letter-spacing: -0.025em !important;
        line-height: 1.15 !important;
    }

    /* ════════════════════════════════════════════
       TAB NAVIGATION
    ════════════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: transparent;
        border-bottom: 1px solid #252840;
        padding: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        border-radius: 0;
        color: #8B8FAA;
        font-weight: 500;
        font-size: 0.88rem;
        padding: 0.65rem 1.4rem;
        transition: color 0.2s, border-color 0.2s;
        margin-bottom: -1px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #C7C4D8;
    }
    .stTabs [aria-selected="true"] {
        background: transparent;
        color: #E2E2EC;
        border-bottom: 2px solid #6C63FF;
        font-weight: 600;
    }

    /* ════════════════════════════════════════════
       CHAT INTERFACE
    ════════════════════════════════════════════ */
    [data-testid="stChatMessage"] {
        background: #1A1D2E;
        border: 1px solid #252840;
        border-radius: 14px;
        margin-bottom: 0.75rem;
        padding: 0.25rem 0;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.15);
        transition: border-color 0.2s;
    }
    [data-testid="stChatMessage"]:hover {
        border-color: rgba(108, 99, 255, 0.2);
    }

    /* Chat input bar */
    [data-testid="stChatInput"] {
        border-top: 1px solid #252840 !important;
        background: #11131A !important;
    }
    [data-testid="stChatInput"] textarea {
        background: #1A1D2E !important;
        border: 1px solid #252840 !important;
        border-radius: 12px !important;
        color: #E2E2EC !important;
        font-size: 0.9rem !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: rgba(108, 99, 255, 0.5) !important;
        box-shadow: 0 0 0 3px rgba(108, 99, 255, 0.1) !important;
    }

    /* ════════════════════════════════════════════
       PAGE HEADINGS & TYPOGRAPHY
    ════════════════════════════════════════════ */
    h1, h2, h3 {
        color: #E2E2EC;
        font-weight: 700;
        letter-spacing: -0.025em;
    }
    h1 { font-size: 1.8rem; }
    h2 { font-size: 1.4rem; font-weight: 600; }
    h3 { font-size: 1.05rem; }

    /* Section subheaders */
    .stApp .stMarkdown h2,
    .stApp .stMarkdown h3 {
        color: #C7C4D8;
    }

    /* ════════════════════════════════════════════
       DIVIDERS
    ════════════════════════════════════════════ */
    hr {
        border: none;
        border-top: 1px solid #252840;
        margin: 2rem 0;
    }

    /* ════════════════════════════════════════════
       EXPANDERS / ACCORDIONS
    ════════════════════════════════════════════ */
    .streamlit-expanderHeader {
        font-size: 0.88rem;
        color: #C7C4D8;
        font-weight: 500;
        background: #1A1D2E !important;
        border: 1px solid #252840 !important;
        border-radius: 10px !important;
        padding: 0.7rem 1rem !important;
        transition: background 0.2s, border-color 0.2s;
    }
    .streamlit-expanderHeader:hover {
        background: #282A31 !important;
        border-color: rgba(108, 99, 255, 0.25) !important;
    }
    .streamlit-expanderContent {
        background: #13151F !important;
        border: 1px solid #252840 !important;
        border-top: none !important;
        border-radius: 0 0 10px 10px !important;
    }

    /* ════════════════════════════════════════════
       ALERT / INFO / WARNING BANNERS
    ════════════════════════════════════════════ */
    .stAlert {
        border-radius: 10px !important;
        border-left-width: 3px !important;
        font-size: 0.875rem;
    }
    [data-testid="stNotificationContentInfo"] {
        background: rgba(108, 99, 255, 0.08) !important;
        border-color: rgba(108, 99, 255, 0.4) !important;
    }
    [data-testid="stNotificationContentWarning"] {
        background: rgba(245, 158, 11, 0.08) !important;
        border-color: rgba(245, 158, 11, 0.4) !important;
    }
    [data-testid="stNotificationContentError"] {
        background: rgba(239, 68, 68, 0.08) !important;
        border-color: rgba(239, 68, 68, 0.4) !important;
    }
    [data-testid="stNotificationContentSuccess"] {
        background: rgba(34, 197, 94, 0.08) !important;
        border-color: rgba(34, 197, 94, 0.4) !important;
    }

    /* ════════════════════════════════════════════
       PROGRESS BAR
    ════════════════════════════════════════════ */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #6C63FF, #8781FF) !important;
        border-radius: 99px;
    }
    .stProgress > div > div {
        background: #252840 !important;
        border-radius: 99px;
    }

    /* ════════════════════════════════════════════
       SPINNER
    ════════════════════════════════════════════ */
    .stSpinner > div {
        border-top-color: #6C63FF !important;
    }

    /* ════════════════════════════════════════════
       IMAGES (video thumbnails)
    ════════════════════════════════════════════ */
    [data-testid="stImage"] img {
        border-radius: 10px;
        border: 1px solid #252840;
    }

    /* ════════════════════════════════════════════
       SCROLLBAR (WebKit)
    ════════════════════════════════════════════ */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0B0D14; }
    ::-webkit-scrollbar-thumb {
        background: #252840;
        border-radius: 99px;
    }
    ::-webkit-scrollbar-thumb:hover { background: #6C63FF; }

    /* ════════════════════════════════════════════
       SELECTION
    ════════════════════════════════════════════ */
    ::selection {
        background: rgba(108, 99, 255, 0.25);
        color: #E2E2EC;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── sidebar brand ─────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
          <div class="brand-icon-wrap">▶</div>
          <div>
            <div class="brand-name">PlaylistIQ</div>
            <div class="brand-tagline">AI-powered YouTube analysis</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── sidebar continued (API keys + input) ─────────────────────────────────
    with st.sidebar:
        # Load API keys from Streamlit secrets
        try:
            yt_key: str = st.secrets["YOUTUBE_API_KEY"]
            g_key: str = st.secrets["GOOGLE_API_KEY"]
        except (KeyError, FileNotFoundError):
            st.error("API keys not configured.")
            st.markdown(
                "Add `YOUTUBE_API_KEY` and `GOOGLE_API_KEY` to your "
                "[Streamlit secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management) "
                "to get started."
            )
            st.stop()

        st.markdown('<div class="sidebar-section-label">Analyze content</div>', unsafe_allow_html=True)

        url_input = st.text_area(
            "Paste a YouTube link",
            placeholder=(
                "Playlist: youtube.com/playlist?list=…\n"
                "Video:    youtube.com/watch?v=…\n"
                "Multiple: url1, url2, url3"
            ),
            height=110,
            key="url_input_box",
            label_visibility="collapsed",
        )

        process_clicked = st.button(
            "Analyze →",
            type="primary",
            disabled=not bool(url_input and url_input.strip()),
        )
        if process_clicked:
            _process_input(url_input, yt_key, g_key)

        if st.session_state.get("videos"):
            st.divider()
            n = len(st.session_state.videos)
            t = len(st.session_state.transcripts)
            st.success(f"{n} videos ready · {t} transcribed")

            if st.button("Start over"):
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

    tab1, tab2 = st.tabs(["Overview", "Chat"])
    with tab1:
        _page_dashboard()
    with tab2:
        _page_chat()


if __name__ == "__main__":
    main()
