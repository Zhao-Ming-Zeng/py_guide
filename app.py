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

# --- 設定 ---
st.set_page_config(page_title="語音導覽", layout="wide", page_icon="🗺️")

# --- CSS 按鈕樣式 ---
st.markdown("""
<style>
    .stButton button {
        background-color: #E63946; color: white; border-radius: 50%;
        width: 80px; height: 80px; font-size: 30px; border: 4px solid white;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.3); margin: 0 auto; display: block;
    }
    .stButton button:hover { background-color: #D62828; transform: scale(1.05); }
    /* 更新定位的小按鈕樣式 */
    div[data-testid="stButton"] button[kind="secondary"] {
        border-radius: 5px; width: auto; height: auto; background-color: #f0f2f6; color: black; font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# --- 載入資料 ---
if not os.path.exists("data/spots.json"):
    st.error("❌ 嚴重錯誤：找不到 data/spots.json，請檢查檔案結構！")
    st.stop()
SPOTS = json.load(open("data/spots.json", "r", encoding="utf-8"))
TRIGGER_DIST = 150

# --- RAG 載入與錯誤診斷 ---
@st.cache_resource
def load_rag():
    # 診斷 1: 檢查索引
    if not os.path.exists("faiss_index"):
        return "MISSING_INDEX"
    
    # 診斷 2: 檢查 Key
    if "GOOGLE_API_KEY" not in st.secrets:
        return "MISSING_KEY"

    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", temperature=0.3, 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        prompt = PromptTemplate.from_template(
            "你是在地導覽員。依據背景回答，不知道就說不知道。\n背景:{context}\n問題:{question}"
        )
        
        return (
            {"context": db.as_retriever(search_kwargs={"k": 2}), "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )
    except Exception as e:
        return f"ERROR: {str(e)}"

qa_chain_or_error = load_rag()

# --- 播放器 ---
def get_player(path):
    if not os.path.exists(path): return None
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'<audio autoplay controls src="data:audio/mp3;base64,{b64}" style="width:100%;"></audio>'

# ================== 主畫面 ==================
st.title("🗺️ 隨身語音導覽")

# --- GPS 強制更新邏輯 ---
if 'gps_key' not in st.session_state:
    st.session_state.gps_key = 0

col_gps_info, col_gps_btn = st.columns([3, 1])
with col_gps_btn:
    if st.button("🔄 更新定位", key="refresh_btn", help="點擊強制重新抓取 GPS"):
        st.session_state.gps_key += 1 # 改變 key 會強制重新掛載元件
        st.rerun()

# 取得定位 (使用動態 Key)
loc = get_geolocation(key=f"gps_{st.session_state.gps_key}")

if loc:
    user_pos = (loc["coords"]["latitude"], loc["coords"]["longitude"])
    
    # --- 顯示地圖 ---
    m = folium.Map(location=user_pos, zoom_start=17)
    folium.Marker(user_pos, popup="我", icon=folium.Icon(color="blue", icon="user")).add_to(m)
    
    nearest_key = None
    min_dist = float("inf")

    for key, info in SPOTS.items():
        spot_pos = (info["lat"], info["lon"])
        d = geodesic(user_pos, spot_pos).meters
        folium.Marker(spot_pos, popup=info["name"], icon=folium.Icon(color="red", icon="info-sign")).add_to(m)
        folium.Circle(spot_pos, radius=TRIGGER_DIST, color="red", fill=True, fill_opacity=0.1).add_to(m)
        
        if d < min_dist:
            min_dist = d
            nearest_key = key

    st_folium(m, width=700, height=350)
    
    # --- 觸發區 ---
    if min_dist <= TRIGGER_DIST:
        spot = SPOTS[nearest_key]
        st.success(f"📍 抵達：{spot['name']} (距離 {int(min_dist)}m)")
        
        lang = st.radio("語言", ["中文", "台語"], horizontal=True)
        st.info(spot["intro_cn"] if lang == "中文" else spot.get("intro_tw", "無台語介紹"))
        
        # 播放
        if st.button("▶"):
            suffix = "cn" if lang == "中文" else "tw"
            path = f"data/audio/{nearest_key}_{suffix}.mp3"
            player = get_player(path)
            if player: st.markdown(player, unsafe_allow_html=True)
            else: st.error(f"⚠️ 找不到音檔：{path} (請先執行 1_gen_assets.py)")

        # --- 問答區 ---
        st.divider()
        q = st.chat_input(f"關於 {spot['name']} 的提問")
        
        if q:
            # 檢查 RAG 狀態
            if isinstance(qa_chain_or_error, str):
                # 這裡處理錯誤，讓使用者知道為什麼沒反應
                if qa_chain_or_error == "MISSING_INDEX":
                    st.error("⚠️ 無法回答：尚未建立索引。請先在電腦執行 `python 2_build_index.py`！")
                elif qa_chain_or_error == "MISSING_KEY":
                    st.error("⚠️ 無法回答：缺少 Google API Key。")
                else:
                    st.error(f"⚠️ 系統錯誤：{qa_chain_or_error}")
            elif qa_chain_or_error:
                # 正常回答
                with st.spinner("AI 思考中..."):
                    ans = qa_chain_or_error.invoke(f"關於 {spot['name']}：{q}")
                    st.write(ans)
    else:
        st.info(f"請移動至紅色範圍內 (最近: {SPOTS[nearest_key]['name']})")
else:
    st.warning("📡 等待 GPS 定位中... (若很久沒反應，請按右上方更新按鈕)")