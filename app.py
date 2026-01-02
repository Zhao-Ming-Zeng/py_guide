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


#測試

# --- Streamlit 設定 ---
st.set_page_config(
    page_title="語音導覽",
    layout="wide",
    page_icon="🗺️"
)

# --- CSS：圓形播放按鈕 ---
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

# --- 載入景點資料 ---
SPOTS = json.load(open("data/spots.json", "r", encoding="utf-8"))
TRIGGER_DIST = 150  # 公尺

# --- RAG（LangChain 1.x 正確寫法） ---
@st.cache_resource
def load_rag():
    if not os.path.exists("faiss_index"):
        return None

    if "GOOGLE_API_KEY" not in st.secrets:
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever(search_kwargs={"k": 2})

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )

    prompt = PromptTemplate.from_template(
        """你是在地導覽員。
請根據背景資料回答問題，
如果資料中沒有答案，請直接說「不知道」。

背景資料：
{context}

問題：
{question}
"""
    )

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

qa_chain = load_rag()

# --- 播放本地 MP3 ---
def get_audio_player(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"""
    <audio autoplay controls style="width:100%;">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """

# ================== 主畫面 ==================
st.title("🗺️ 隨身語音導覽")

loc = get_geolocation()

if loc:
    user_pos = (
        loc["coords"]["latitude"],
        loc["coords"]["longitude"]
    )

    # --- 地圖 ---
    m = folium.Map(location=user_pos, zoom_start=17)
    folium.Marker(
        user_pos,
        popup="我",
        icon=folium.Icon(color="blue", icon="user")
    ).add_to(m)

    nearest_key = None
    min_dist = float("inf")

    for key, info in SPOTS.items():
        spot_pos = (info["lat"], info["lon"])
        d = geodesic(user_pos, spot_pos).meters

        folium.Marker(
            spot_pos,
            popup=info["name"],
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

    st_folium(m, width=700, height=350)

    # --- 觸發邏輯 ---
    if min_dist <= TRIGGER_DIST:
        spot = SPOTS[nearest_key]
        spot_name = spot["name"]

        st.success(f"📍 抵達：{spot_name}（距離 {int(min_dist)} m）")

        # 語言選擇
        lang = st.radio(
            "選擇語言",
            ["中文", "台語"],
            horizontal=True
        )

        intro = (
            spot["intro_cn"]
            if lang == "中文"
            else spot.get("intro_tw", "暫無台語介紹")
        )
        st.info(intro)

        # 播放按鈕
        if st.button("▶"):
            suffix = "cn" if lang == "中文" else "tw"
            audio_path = f"data/audio/{nearest_key}_{suffix}.mp3"
            player = get_audio_player(audio_path)
            if player:
                st.markdown(player, unsafe_allow_html=True)
            else:
                st.error("找不到語音檔案")

        # --- RAG 問答 ---
        st.divider()
        q = st.chat_input(f"關於 {spot_name} 的提問")

        if q and qa_chain:
            with st.spinner("AI 思考中..."):
                answer = qa_chain.invoke(
                    f"關於 {spot_name}：{q}"
                )
                st.write(answer)

    else:
        st.info(
            f"請移動至紅色範圍內（最近景點：{SPOTS[nearest_key]['name']}）"
        )

else:
    st.warning("📡 等待 GPS 定位中...")
