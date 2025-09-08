import streamlit as st
from pathlib import Path
import json

VIDEO_DIR = Path("/projectnb/ivc-ml/xthomas/SHARED/video_evals/YOUTUBE_DATA")
SAVE_DIR = Path("./scores")
SAVE_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Video Evaluation", layout="wide")
st.markdown("""
    <style>
      .block-container { padding-top: 2rem; max-width: 900px; }
      h1, h2, h3 { letter-spacing: -0.02em; margin-bottom: 0.5rem; }
      .stTextInput, .stSelectbox, .stNumberInput, .stRadio, .stButton>button {
        background-color: transparent !important;
        box-shadow: none !important;
      }
      div[data-testid="stForm"] {
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
      }
      .video-name {
        font-family: monospace;
        font-size: 13px;
        color: #6b7280;
        margin-bottom: 10px;
      }
      div[role="radiogroup"] > label {
        border: 1px solid #d1d5db !important;
        border-radius: 10px !important;
        padding: 8px 14px !important;
        margin-right: 6px !important;
      }
    </style>
""", unsafe_allow_html=True)

st.title("Video Evaluation")
username = st.text_input("Username (same each time to resume):", key="username").strip()
if not username:
    st.stop()

save_file = SAVE_DIR / f"{username}_scores.json"

if save_file.exists():
    with open(save_file, "r") as f:
        scores = json.load(f)
else:
    scores = {}

video_files = sorted(VIDEO_DIR.glob("*.mp4"))
if not video_files:
    st.error(f"No .mp4 videos found in {VIDEO_DIR}")
    st.stop()
video_names = [v.name for v in video_files]

def set_selected_for_current():
    name = video_files[st.session_state.current_index].name
    st.session_state.current_video_name = name
    radio_key = f"score_radio::{name}"
    saved = scores.get(name, None)
    st.session_state.pop(radio_key, None)
    if saved is not None:
        st.session_state[radio_key] = saved
    st.session_state.selected_score = saved

if "current_index" not in st.session_state:
    unscored = [i for i, n in enumerate(video_names) if n not in scores]
    st.session_state.current_index = unscored[0] if unscored else 0
    set_selected_for_current()

current_idx = int(st.session_state.current_index)
current_video = video_files[current_idx]
if st.session_state.get("current_video_name") != current_video.name:
    set_selected_for_current()

st.markdown(
    """
**You will see AI-generated videos of the following actions:**

JumpingJack, PullUps, PushUps, HulaHoop, WallPushups, Shotput, SoccerJuggling, TennisSwing, ThrowDiscus, BodyWeightSquats.

**Give one score (1–5)** for each video based on how well the human is performing the intended action overall.

**Consider:**
- Whether the movement matches the intended action
- How clearly and smoothly it is carried out
"""
)

st.subheader(f"Video {current_idx + 1} / {len(video_files)}")
st.markdown(f'<div class="video-name">{current_video.name}</div>', unsafe_allow_html=True)
st.video(str(current_video))

radio_key = f"score_radio::{current_video.name}"
with st.form("score_form", clear_on_submit=False):
    st.write("**Select a score (1-5):**")
    selected = st.radio(
        "Score",
        options=[1, 2, 3, 4, 5],
        index=None,
        horizontal=True,
        label_visibility="collapsed",
        key=radio_key,
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        prev_clicked = st.form_submit_button("◀ Previous", use_container_width=True)
    with c2:
        save_next_clicked = st.form_submit_button("Save & Next ▶", use_container_width=True)

st.session_state.selected_score = st.session_state.get(radio_key)

if save_next_clicked:
    if st.session_state.selected_score is None:
        st.warning("Please select a score (1-5) before continuing.")
    else:
        scores[current_video.name] = int(st.session_state.selected_score)
        with open(save_file, "w") as f:
            json.dump(scores, f, indent=2)
        if current_idx < len(video_files) - 1:
            st.session_state.current_index = current_idx + 1
            set_selected_for_current()
            st.rerun()
        else:
            st.success("All videos scored!")

if prev_clicked and current_idx > 0:
    st.session_state.current_index = current_idx - 1
    set_selected_for_current()
    st.rerun()

scored_count = len(scores)
st.write(f"Progress: {scored_count}/{len(video_files)} scored")
st.progress(scored_count / len(video_files))