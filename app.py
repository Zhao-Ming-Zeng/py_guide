import streamlit as st
import json
import time
import math
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium
from streamlit_js_eval import get_geolocation

# ===== RAG =====
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# ======================
# 基本設定
# ======================
ENTER_RADIUS = 120
EXIT_RADIUS = 170
MAP_LIMIT_RADIUS = 300
SIM_THRESHOLD = 0.35

st.set_page_config(layout="wide")
st.title("📍 AI 導覽系統（GPS + RAG）")

# ======================
# Session State
# ======================
if "last_pos" not in st.session_state:
    st.session_state.last_pos = None
if "last_update" not in st.session_state:
    st.session_state.last_update = 0
if "current_spot" not in st.session_state:
    st.session_state.current_spot = None
if "played_spot" not in st.session_state:
    st.session_state.played_spot = None

# ======================
# 載入資料
# ======================
with open("data/spots.json", encoding="utf-8") as f:
    spots = json.load(f)

# ======================
# GPS 節流
# ======================
pos = get_geolocation()
now = time.time()

if pos:
    lat, lon = pos["coords"]["latitude"], pos["coords"]["longitude"]

    update_interval = 10
    if st.session_state.last_pos:
        dist_move = geodesic(st.session_state.last_pos, (lat, lon)).meters
        if dist_move > 8:
            update_interval = 5

    if now - st.session_state.last_update >= update_interval:
        st.session_state.last_pos = (lat, lon)
        st.session_state.last_update = now
else:
    st.warning("⚠️ 尚未取得 GPS")
    st.stop()

# ======================
# 找最近景點
# ======================
nearest = None
nearest_dist = 999999

for k, s in spots.items():
    d = geodesic((lat, lon), (s["lat"], s["lon"])).meters
    if d < nearest_dist:
        nearest, nearest_dist = k, d

# ======================
# 去抖動判斷
# ======================
if st.session_state.current_spot is None:
    if nearest_dist <= ENTER_RADIUS:
        st.session_state.current_spot = nearest
else:
    cur = st.session_state.current_spot
    d_cur = geodesic(
        (lat, lon),
        (spots[cur]["lat"], spots[cur]["lon"])
    ).meters

    if d_cur >= EXIT_RADIUS:
        st.session_state.current_spot = None
        st.session_state.played_spot = None

# ======================
# 地圖
# ======================
m = folium.Map(location=(lat, lon), zoom_start=17)

folium.Marker(
    (lat, lon),
    tooltip="你的位置",
    icon=folium.Icon(color="blue")
).add_to(m)

for k, s in spots.items():
    d = geodesic((lat, lon), (s["lat"], s["lon"])).meters
    if d <= MAP_LIMIT_RADIUS:
        folium.Marker(
            (s["lat"], s["lon"]),
            tooltip=f"{s['name']} ({int(d)}m)"
        ).add_to(m)

st_folium(m, height=400)

# ======================
# 自動播放導覽（無播放條）
# ======================
if st.session_state.current_spot:
    spot = st.session_state.current_spot
    info = spots[spot]
    st.success(f"🎧 你正在 {info['name']} 附近（{int(nearest_dist)} 公尺）")

    if st.session_state.played_spot != spot:
        audio_path = f"data/audio/{spot}_cn.mp3"
        with open(audio_path, "rb") as f:
            b64 = f.read().hex()

        st.markdown(
            f"""
            <audio autoplay hidden>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """,
            unsafe_allow_html=True
        )

        st.session_state.played_spot = spot
else:
    st.info(f"🚶 尚未進入景點（最近距離 {int(nearest_dist)}m）")

# ======================
# AI 問答（RAG 防亂掰）
# ======================
st.divider()
st.subheader("🤖 AI 導覽問答")

query = st.text_input("請輸入你的問題")

if query:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)

    docs_scores = db.similarity_search_with_score(query, k=3)
    best_score = docs_scores[0][1]

    if best_score > SIM_THRESHOLD:
        st.warning("⚠️ 這個問題超出目前導覽資料範圍")
    else:
        context = "\n".join(d.page_content for d, _ in docs_scores)
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-1.5-pro",
            temperature=0.3
        )
        answer = llm.invoke(
            f"根據以下資料回答問題，不要自行推測：\n{context}\n\n問題：{query}"
        )
        st.write(answer.content)
