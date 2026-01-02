import streamlit as st
import json
import time
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium
from streamlit_js_eval import get_geolocation
from streamlit_autorefresh import st_autorefresh

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# ======================
# 基本參數
# ======================
ENTER_RADIUS = 120
EXIT_RADIUS = 170
MAP_LIMIT_RADIUS = 300
SIM_THRESHOLD = 0.35
AUTO_REFRESH_SEC = 5

st.set_page_config(layout="wide")
st.title("📍 AI GPS 導覽系統")

# ======================
# 自動刷新（GPS）
# ======================
st_autorefresh(interval=AUTO_REFRESH_SEC * 1000, key="gps_refresh")

# ======================
# Session State
# ======================
for k, v in {
    "last_pos": None,
    "current_spot": None,
    "played_spot": None,
    "force_refresh": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ======================
# 手動刷新 GPS
# ======================
if st.button("🔄 重新刷新定位"):
    st.session_state.force_refresh = True

# ======================
# 取得 GPS
# ======================
pos = get_geolocation()

if not pos:
    st.warning("⚠️ 等待 GPS 定位中...")
    st.stop()

lat, lon = pos["coords"]["latitude"], pos["coords"]["longitude"]

if st.session_state.force_refresh:
    st.session_state.last_pos = None
    st.session_state.force_refresh = False

# ======================
# 載入景點資料
# ======================
with open("data/spots.json", encoding="utf-8") as f:
    spots = json.load(f)

# ======================
# 找最近景點
# ======================
nearest, nearest_dist = None, 999999
for k, s in spots.items():
    d = geodesic((lat, lon), (s["lat"], s["lon"])).meters
    if d < nearest_dist:
        nearest, nearest_dist = k, d

# ======================
# GPS 去抖動
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
# 自動語音導覽（無播放條）
# ======================
if st.session_state.current_spot:
    spot = st.session_state.current_spot
    info = spots[spot]
    st.success(f"🎧 已進入 {info['name']}（{int(nearest_dist)}m）")

    if st.session_state.played_spot != spot:
        with open(f"data/audio/{spot}_cn.mp3", "rb") as f:
            audio_b64 = f.read().hex()

        st.markdown(
            f"""
            <audio autoplay hidden>
                <source src="data:audio/mp3;base64,{audio_b64}">
            </audio>
            """,
            unsafe_allow_html=True
        )
        st.session_state.played_spot = spot
else:
    st.info(f"🚶 尚未進入景點（最近 {int(nearest_dist)}m）")

# ======================
# AI 問答（有送出鍵）
# ======================
st.divider()
st.subheader("🤖 AI 導覽問答")

with st.form("ai_form"):
    query = st.text_input("請輸入問題")
    submitted = st.form_submit_button("送出提問")

if submitted and query:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(
        "faiss_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs_scores = db.similarity_search_with_score(query, k=3)

    if not docs_scores or docs_scores[0][1] > SIM_THRESHOLD:
        st.warning("⚠️ 這個問題超出目前導覽資料範圍")
    else:
        context = "\n".join(d.page_content for d, _ in docs_scores)

        llm = ChatGoogleGenerativeAI(
            model="models/gemini-1.0-pro",
            temperature=0.3
        )

        answer = llm.invoke(
            f"請只根據以下資料回答，不要自行推測：\n{context}\n\n問題：{query}"
        )
        st.write(answer.content)
