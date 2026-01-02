import streamlit as st
import json
import os
import base64
import folium

from streamlit_folium import st_folium
from streamlit_js_eval import get_geolocation
from geopy.distance import geodesic

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ======================================================
# 1️⃣ 頁面設定
# ======================================================
st.set_page_config(
    page_title="🗺️ 雲科大語音導覽",
    layout="wide",
    page_icon="🗺️"
)

# ======================================================
# 2️⃣ CSS 美化
# ======================================================
st.markdown("""
<style>
.stButton button {
    background-color: #E63946;
    color: white;
    border-radius: 50%;
    width: 80px;
    height: 80px;
    font-size: 30px;
    border: 4px solid white;
    box-shadow: 0px 4px 8px rgba(0,0,0,0.3);
    margin: 0 auto;
    display: block;
}
.stButton button:hover {
    background-color: #D62828;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# 3️⃣ 載入景點資料
# ======================================================
json_path = "data/spots.json"
if not os.path.exists(json_path):
    st.error("❌ 找不到 data/spots.json")
    st.stop()

with open(json_path, "r", encoding="utf-8") as f:
    SPOTS = json.load(f)

TRIGGER_DIST = 150  # 公尺

# ======================================================
# 4️⃣ 載入 RAG（含防呆）
# ======================================================
@st.cache_resource
def load_rag():
    if not os.path.exists("faiss_index"):
        return "MISSING_INDEX"
    if "GOOGLE_API_KEY" not in st.secrets:
        return "MISSING_KEY"

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",  # ✅ 修正模型
            temperature=0.3,
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )

        prompt = PromptTemplate.from_template(
            "你是一位在地校園導覽員，只能根據下列背景資訊回答。\n"
            "若背景中沒有答案，請直接說不知道。\n\n"
            "背景資訊：{context}\n"
            "問題：{question}"
        )

        chain = (
            {
                "context": db.as_retriever(search_kwargs={"k": 2}),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain

    except Exception as e:
        return f"ERROR: {e}"

qa_chain = load_rag()

# ======================================================
# 5️⃣ 音檔播放器
# ======================================================
def get_player(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        audio_bytes = f.read()
    b64 = base64.b64encode(audio_bytes).decode()
    return f"""
    <audio autoplay controls style="width:100%">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """

# ======================================================
# 6️⃣ 主畫面
# ======================================================
st.title("🗺️ 雲科大隨身語音導覽")

# GPS 重新整理
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🔄 更新定位"):
        st.rerun()

# ✅ 正確呼叫（不能帶 key）
loc = get_geolocation()

if not loc:
    st.warning("📡 正在取得 GPS 定位，請允許瀏覽器定位權限")
    st.stop()

user_pos = (
    loc["coords"]["latitude"],
    loc["coords"]["longitude"]
)

# ======================================================
# 7️⃣ 地圖與距離計算
# ======================================================
m = folium.Map(location=user_pos, zoom_start=17)

folium.Marker(
    user_pos,
    popup="你的位置",
    icon=folium.Icon(color="blue", icon="user")
).add_to(m)

nearest_key = None
min_dist = float("inf")

for key, spot in SPOTS.items():
    spot_pos = (spot["lat"], spot["lon"])
    d = geodesic(user_pos, spot_pos).meters

    folium.Marker(
        spot_pos,
        popup=f"{spot['name']} ({int(d)}m)",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)

    folium.Circle(
        spot_pos,
        radius=TRIGGER_DIST,
        color="red",
        fill=True,
        fill_opacity=0.1
    ).add_to(m)

    if d < min_dist:
        min_dist = d
        nearest_key = key

with col1:
    st_folium(m, height=350, width=700)

# ======================================================
# 8️⃣ 進入景點範圍
# ======================================================
if nearest_key and min_dist <= TRIGGER_DIST:
    spot = SPOTS[nearest_key]

    st.success(f"📍 已抵達 **{spot['name']}**（{int(min_dist)} 公尺）")

    lang = st.radio("語音導覽語言", ["中文", "台語"], horizontal=True)

    intro_text = (
        spot["intro_cn"]
        if lang == "中文"
        else spot.get("intro_tw", "（暫無台語文字）")
    )
    st.info(intro_text)

    if st.button("▶ 播放語音導覽"):
        suffix = "cn" if lang == "中文" else "tw"
        audio_path = f"data/audio/{nearest_key}_{suffix}.mp3"
        player = get_player(audio_path)
        if player:
            st.markdown(player, unsafe_allow_html=True)
        else:
            st.warning("⚠️ 音檔尚未生成")

    # ==================================================
    # 9️⃣ AI + RAG 問答（完整補回）
    # ==================================================
    st.divider()
    st.markdown(f"### 💬 詢問 AI 導覽員（{spot['name']}）")

    user_q = st.chat_input("例如：這棟建築的歷史是什麼？")

    if user_q:
        if isinstance(qa_chain, str):
            if qa_chain == "MISSING_INDEX":
                st.error("❌ 尚未建立 FAISS 索引")
            elif qa_chain == "MISSING_KEY":
                st.error("❌ 缺少 GOOGLE_API_KEY")
            else:
                st.error(qa_chain)
        else:
            with st.chat_message("user"):
                st.write(user_q)

            with st.chat_message("assistant"):
                with st.spinner("AI 導覽員思考中..."):
                    q = f"我現在在「{spot['name']}」，{user_q}"
                    answer = qa_chain.invoke(q)
                    st.write(answer)

else:
    st.info("🚶 尚未進入任何景點導覽範圍")
