import streamlit as st
import sys
import os
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.config import (
    ANTHROPIC_API_KEY, YOUTUBE_API_KEY, LLM_MODEL,
    LLM_TEMPERATURE, RETRIEVER_K, MEMORY_WINDOW_SIZE,
)
from app.components.styles import get_css

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="YT Knowledge Assistant",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(get_css(), unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────
def _init():
    defaults = {
        'session_id':        str(uuid.uuid4()),
        'messages':          [],
        'loaded_videos':     {},
        'vector_store':      None,
        'metadata_store':    None,
        'llm':               None,
        'embeddings':        None,
        'retriever':         None,
        'chat_history_text': "",
        'output_format':     "Auto",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ── Cached initialisers ────────────────────────────────────
@st.cache_resource
def init_llm(api_key: str):
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=LLM_MODEL, temperature=LLM_TEMPERATURE,
                         anthropic_api_key=api_key)

@st.cache_resource
def init_embeddings():
    from core.processing.embedder import get_embedding_model
    return get_embedding_model(provider="huggingface")

@st.cache_resource
def init_stores(_emb):
    from core.retrieval.vector_store import VectorStoreManager
    from core.retrieval.metadata_store import MetadataStore
    return VectorStoreManager(embedding_model=_emb), MetadataStore()


# ── Constants ──────────────────────────────────────────────
INTENT_META = {
    "factual_qa":          ("Q&A",        "intent-qa"),
    "summarize":           ("Summary",    "intent-summarize"),
    "compare_videos":      ("Compare",    "intent-compare"),
    "sentiment_query":     ("Sentiment",  "intent-sentiment"),
    "generate_flashcards": ("Flashcards", "intent-flashcards"),
    "generate_quiz":       ("Quiz",       "intent-quiz"),
}

FORMAT_OPTIONS = {
    "Auto":           "Auto — let the AI decide",
    "Detailed Answer":"Detailed Q&A answer",
    "Summary":        "Structured summary",
    "Study Notes":    "Study notes",
    "Flashcards":     "Flashcard set",
    "Quiz":           "Multiple-choice quiz",
    "Compare Videos": "Cross-video comparison",
}

FORMAT_TO_INTENT = {
    "Detailed Answer": "factual_qa",
    "Summary":         "summarize",
    "Study Notes":     "summarize",
    "Flashcards":      "generate_flashcards",
    "Quiz":            "generate_quiz",
    "Compare Videos":  "compare_videos",
}


def total_chunks():
    return sum(v.get('chunks', 0) for v in st.session_state.loaded_videos.values())


# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:

    # Brand
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-icon">▶</div>
        <div>
            <p class="sb-logo-title">YT Knowledge</p>
            <p class="sb-logo-sub">claude · rag · chromadb</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # API Keys
    st.markdown('<div class="sb-section-label">API Keys</div>', unsafe_allow_html=True)
    with st.expander("Configure keys", expanded=not bool(ANTHROPIC_API_KEY)):
        anthropic_key = st.text_input(
            "Anthropic API Key",
            value=ANTHROPIC_API_KEY,
            type="password",
            placeholder="sk-ant-...",
            help="Required — powers Claude",
        )
        yt_key = st.text_input(
            "YouTube API Key",
            value=YOUTUBE_API_KEY,
            type="password",
            placeholder="AIza...",
            help="Optional — comments & metadata",
        )

    if anthropic_key:
        st.markdown('<div class="api-status ok">✓ anthropic key loaded</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="api-status warn">⚠ anthropic key required</div>', unsafe_allow_html=True)

    st.divider()

    # Add Video
    st.markdown('<div class="sb-section-label">Add Video</div>', unsafe_allow_html=True)
    url_input = st.text_input(
        "url",
        placeholder="youtube.com/watch?v=...  or  playlist...",
        label_visibility="collapsed",
    )
    process_btn = st.button(
        "▶  Process Video",
        type="primary",
        use_container_width=True,
        disabled=not (url_input.strip() and anthropic_key),
    )

    st.divider()

    # Library
    if st.session_state.loaded_videos:
        st.markdown(
            f'<div class="sb-section-label">Library '
            f'<span class="count-badge">{len(st.session_state.loaded_videos)}</span></div>',
            unsafe_allow_html=True,
        )
        for vid, info in st.session_state.loaded_videos.items():
            thumb   = info.get('thumbnail') or f"https://img.youtube.com/vi/{vid}/mqdefault.jpg"
            title   = info.get('title', vid)[:38]
            chunks  = info.get('chunks', 0)
            channel = info.get('channel', '')[:20]
            st.markdown(f"""
            <div class="video-card-sb">
                <img class="video-thumb" src="{thumb}"
                     onerror="this.src='https://img.youtube.com/vi/{vid}/mqdefault.jpg'"/>
                <div class="video-info">
                    <div class="video-title-sb" title="{info.get('title', vid)}">{title}</div>
                    <div class="video-meta">
                        <span class="badge badge-amber">{chunks} chunks</span>
                        {"<span class='badge badge-teal'>" + channel + "</span>" if channel else ""}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Clear All", type="secondary", use_container_width=True):
                if st.session_state.vector_store:
                    st.session_state.vector_store.clear_all()
                st.session_state.loaded_videos     = {}
                st.session_state.messages          = []
                st.session_state.chat_history_text = ""
                st.rerun()
        with col_b:
            if st.session_state.messages:
                if st.button("Clear Chat", type="secondary", use_container_width=True):
                    st.session_state.messages          = []
                    st.session_state.chat_history_text = ""
                    st.rerun()

        st.divider()

    # Format selector — radio list
    st.markdown('<div class="sb-section-label">Response Format</div>', unsafe_allow_html=True)
    fmt_keys   = list(FORMAT_OPTIONS.keys())
    fmt_labels = list(FORMAT_OPTIONS.values())
    cur_idx    = fmt_keys.index(st.session_state.output_format) if st.session_state.output_format in fmt_keys else 0
    chosen_label = st.radio(
        "format",
        options=fmt_labels,
        index=cur_idx,
        label_visibility="collapsed",
    )
    chosen_key = fmt_keys[fmt_labels.index(chosen_label)]
    if chosen_key != st.session_state.output_format:
        st.session_state.output_format = chosen_key
        st.rerun()

    st.divider()

    # Tech stack
    st.markdown('<div class="sb-section-label">Tech Stack</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="model-info-card">
        <div class="model-row">
            <span class="model-label">LLM</span>
            <span class="model-value">{LLM_MODEL}</span>
        </div>
        <div class="model-row">
            <span class="model-label">Embeddings</span>
            <span class="model-value">all-MiniLM-L6-v2</span>
        </div>
        <div class="model-row">
            <span class="model-label">Vector DB</span>
            <span class="model-value">ChromaDB</span>
        </div>
        <div class="model-row">
            <span class="model-label">Retrieval</span>
            <span class="model-value">MMR · K={RETRIEVER_K}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Main: Hero (left-aligned) ──────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">RAG &nbsp;·&nbsp; Multi-Video &nbsp;·&nbsp; Timestamped</div>
    <h1 class="hero-title">
        YouTube<br><span class="hl">Knowledge</span> Assistant
    </h1>
    <p class="hero-sub">
        Chat with any YouTube video and get accurate, timestamped answers —
        plus summaries, flashcards, quizzes, and cross-video comparisons.
    </p>
</div>
""", unsafe_allow_html=True)

# Stats strip
vcount = len(st.session_state.loaded_videos)
ccount = total_chunks()
mcount = len(st.session_state.messages) // 2
fmt    = st.session_state.output_format

st.markdown(f"""
<div class="stats-strip">
    <div class="stat-item">
        <div class="stat-dot amber"></div>
        <div class="stat-text">
            <span class="stat-number">{vcount}</span>
            <span class="stat-desc">Videos</span>
        </div>
    </div>
    <div class="stat-item">
        <div class="stat-dot teal"></div>
        <div class="stat-text">
            <span class="stat-number">{ccount}</span>
            <span class="stat-desc">Chunks</span>
        </div>
    </div>
    <div class="stat-item">
        <div class="stat-dot sky"></div>
        <div class="stat-text">
            <span class="stat-number">{mcount}</span>
            <span class="stat-desc">Exchanges</span>
        </div>
    </div>
    <div class="stat-item">
        <div class="stat-dot emerald"></div>
        <div class="stat-text">
            <span class="stat-number-sm">{fmt}</span>
            <span class="stat-desc">Format</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Video processing ───────────────────────────────────────
if process_btn and url_input and anthropic_key:
    with st.status("Processing video…", expanded=True) as status:
        try:
            st.write("Initialising models…")
            llm = init_llm(anthropic_key)
            embeddings = init_embeddings()
            vector_store, metadata_store = init_stores(embeddings)
            st.session_state.llm            = llm
            st.session_state.vector_store   = vector_store
            st.session_state.metadata_store = metadata_store

            st.write("Resolving URL…")
            from core.ingestion.url_parser import resolve_video_ids
            videos = resolve_video_ids(url_input, yt_key)
            if not videos:
                st.error("Could not resolve any videos from that URL.")
                status.update(label="Failed — no videos found.", state="error")
                st.stop()

            for video_info in videos:
                vid = video_info.video_id
                st.write(f"Fetching transcript: **{video_info.title or vid}**")
                from core.ingestion.transcript import fetch_transcript
                transcript = fetch_transcript(vid)
                if not transcript:
                    st.warning(f"No transcript for {vid} — skipping.")
                    continue

                st.write("Chunking & embedding…")
                from core.processing.chunker import chunk_transcript, chunks_to_documents
                title     = video_info.title or vid
                chunks    = chunk_transcript(transcript, video_title=title)
                documents = chunks_to_documents(chunks)

                st.write("Saving to vector store…")
                vector_store.add_documents(documents)
                metadata_store.save_video(
                    video_id=vid, title=title,
                    channel=video_info.channel,
                    duration=video_info.duration,
                    thumbnail_url=video_info.thumbnail_url,
                    transcript_source=transcript.source,
                    chunk_count=len(chunks),
                )

                if yt_key:
                    st.write("Analysing comments & sentiment…")
                    from core.ingestion.comments import fetch_comments
                    from core.processing.sentiment import run_full_analysis
                    comments = fetch_comments(vid, yt_key)
                    if comments:
                        sentiment = run_full_analysis(vid, comments, llm)
                        metadata_store.save_sentiment(vid, sentiment)

                st.session_state.loaded_videos[vid] = {
                    'title':     title,
                    'thumbnail': video_info.thumbnail_url,
                    'chunks':    len(chunks),
                    'channel':   video_info.channel,
                }

            st.write("Building retriever…")
            from core.retrieval.retriever import get_multi_query_retriever
            st.session_state.retriever = get_multi_query_retriever(
                vector_store, llm, search_type="mmr", k=RETRIEVER_K)

            n = len(st.session_state.loaded_videos)
            status.update(label=f"Done — {n} video(s) indexed. Start chatting!", state="complete")
            st.rerun()

        except Exception as e:
            status.update(label="Processing failed.", state="error")
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())


# ── Welcome screen ─────────────────────────────────────────
if not st.session_state.loaded_videos:
    st.markdown("""
    <div class="welcome-wrap">
        <div class="welcome-title">No videos loaded yet</div>
        <div class="welcome-sub">
            Paste a YouTube URL in the sidebar and click <strong style="color:#f59e0b">▶ Process Video</strong>.<br>
            Supports individual videos, playlists, and channels.
        </div>
        <div class="feature-grid">
            <div class="feature-card">
                <span class="fc-num">01</span>
                <span class="fc-icon">⏱</span>
                <div class="fc-title">Timestamped Answers</div>
                <div class="fc-desc">Every answer links back to the exact moment in the video.</div>
            </div>
            <div class="feature-card">
                <span class="fc-num">02</span>
                <span class="fc-icon">⚖️</span>
                <div class="fc-title">Cross-Video Compare</div>
                <div class="fc-desc">Load multiple videos and ask comparative questions.</div>
            </div>
            <div class="feature-card">
                <span class="fc-num">03</span>
                <span class="fc-icon">❤️</span>
                <div class="fc-title">Sentiment Analysis</div>
                <div class="fc-desc">Understand how the community reacted to the content.</div>
            </div>
            <div class="feature-card">
                <span class="fc-num">04</span>
                <span class="fc-icon">🎴</span>
                <div class="fc-title">Flashcards</div>
                <div class="fc-desc">Auto-generate study flashcards from video content.</div>
            </div>
            <div class="feature-card">
                <span class="fc-num">05</span>
                <span class="fc-icon">🧩</span>
                <div class="fc-title">Quizzes</div>
                <div class="fc-desc">Test your knowledge with AI-generated MCQ quizzes.</div>
            </div>
            <div class="feature-card">
                <span class="fc-num">06</span>
                <span class="fc-icon">🧠</span>
                <div class="fc-title">Memory</div>
                <div class="fc-desc">Conversational context persists across follow-up questions.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Chat history ───────────────────────────────────────
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="◉"):
                st.markdown(msg["content"])
        else:
            intent = msg.get("intent", "")
            badge_label, badge_cls = INTENT_META.get(intent, ("", ""))
            with st.chat_message("assistant", avatar="▶"):
                if badge_label:
                    st.markdown(
                        f'<span class="intent-badge {badge_cls}">{badge_label}</span>',
                        unsafe_allow_html=True,
                    )
                st.markdown(msg["content"])


# ── Chat input ─────────────────────────────────────────────
prompt = st.chat_input(
    "Ask anything about the video(s)…",
    disabled=not st.session_state.loaded_videos,
)

if prompt and st.session_state.loaded_videos:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="◉"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="▶"):
        with st.spinner("Thinking…"):
            try:
                llm          = st.session_state.llm
                retriever    = st.session_state.retriever
                vs           = st.session_state.vector_store
                ms           = st.session_state.metadata_store
                video_ids    = list(st.session_state.loaded_videos.keys())
                video_titles = {v: i['title'] for v, i in st.session_state.loaded_videos.items()}
                titles_str   = ", ".join(video_titles.values())
                fmt          = st.session_state.output_format

                # Route intent
                if fmt == "Auto":
                    from core.chains.router import classify_intent
                    intent = classify_intent(llm, prompt, len(video_ids), titles_str)
                else:
                    intent = FORMAT_TO_INTENT.get(fmt, "factual_qa")

                # Execute chain
                if intent == "factual_qa":
                    from core.chains.qa_chain import run_qa
                    result   = run_qa(llm, retriever, prompt, st.session_state.chat_history_text)
                    response = result["answer"]

                elif intent == "summarize":
                    from core.chains.summary_chain import run_summary
                    style    = "study_notes" if fmt == "Study Notes" else "detailed"
                    response = run_summary(llm, retriever, style=style)

                elif intent == "compare_videos":
                    from core.chains.compare_chain import run_comparison
                    if len(video_ids) < 2:
                        response = "I need at least **2 videos** loaded to compare. Add another video in the sidebar."
                    else:
                        response = run_comparison(llm, vs, prompt, video_ids, video_titles)

                elif intent == "sentiment_query":
                    from core.chains.sentiment_chain import run_sentiment_query
                    sentiment_data = {}
                    for v in video_ids:
                        s = ms.get_sentiment(v)
                        if s:
                            sentiment_data = s
                            break
                    response = run_sentiment_query(llm, retriever, prompt, sentiment_data)

                elif intent == "generate_flashcards":
                    from core.chains.formatter import generate_flashcards, format_flashcards_markdown
                    cards    = generate_flashcards(llm, retriever, topic=prompt)
                    response = format_flashcards_markdown(cards)

                elif intent == "generate_quiz":
                    from core.chains.formatter import generate_quiz, format_quiz_markdown
                    quiz     = generate_quiz(llm, retriever, topic=prompt)
                    response = format_quiz_markdown(quiz)

                else:
                    from core.chains.qa_chain import run_qa
                    result   = run_qa(llm, retriever, prompt, st.session_state.chat_history_text)
                    response = result["answer"]

                # Display
                badge_label, badge_cls = INTENT_META.get(intent, ("", ""))
                if badge_label:
                    st.markdown(
                        f'<span class="intent-badge {badge_cls}">{badge_label}</span>',
                        unsafe_allow_html=True,
                    )
                st.markdown(response)

                st.session_state.messages.append({
                    "role": "assistant", "content": response, "intent": intent,
                })

                recent = st.session_state.messages[-(MEMORY_WINDOW_SIZE * 2):]
                st.session_state.chat_history_text = "\n".join(
                    f"{'Human' if m['role']=='user' else 'Assistant'}: {m['content'][:200]}"
                    for m in recent
                )

                if ms:
                    ms.save_chat_message(st.session_state.session_id, "user", prompt)
                    ms.save_chat_message(st.session_state.session_id, "assistant", response[:500])

            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())


# ── Footer ─────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    youtube knowledge assistant &nbsp;·&nbsp;
    <span class="hl-a">claude</span> &nbsp;·&nbsp;
    <span class="hl-t">langchain</span> &nbsp;·&nbsp;
    <span class="hl-s">chromadb</span> &nbsp;·&nbsp;
    <span class="hl-e">streamlit</span>
</div>
""", unsafe_allow_html=True)
