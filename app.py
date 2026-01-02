import streamlit as st
import json
import os
import base64
import time
import folium
from streamlit_folium import st_folium
from streamlit_js_eval import get_geolocation
from geopy.distance import geodesic
# ---------------------------------------------------------
# 模型相關 (保持原樣)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# ---------------------------------------------------------

st.set_page_config(page_title="語音導覽", layout="wide", page_icon="🗺️")

# --- CSS ---
st.markdown("""
<style>
    .stButton button {
        background-color: #E63946; color: white; border-radius: 50%;
        width: 80px; height: 80px; font-size: 30px; border: 4px solid white;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.3); margin: 0 auto; display: block;
    }
    div[data-testid="stVerticalBlock"] > div > div[data-testid="stButton"] > button {
        width: auto; height: auto; border-radius: 5px; font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# --- 載入資料 ---
if not os.path.exists("data/spots.json"):
    st.error("❌ 找不到 data/spots.json")
    st.stop()
SPOTS = json.load(open("data/spots.json", "r", encoding="utf-8"))
TRIGGER_DIST = 150

# --- RAG ---
@st.cache_resource
def load_rag():
    if not os.path.exists("faiss_index"): return "MISSING_INDEX"
    if "GOOGLE_API_KEY" not in st.secrets: return "MISSING_KEY"
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=st.secrets["GOOGLE_API_KEY"])
        prompt = PromptTemplate.from_template("背景:{context}\n問題:{question}\n回答:")
        return ({"context": db.as_retriever(search_kwargs={"k": 2}), "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    except Exception as e: return str(e)

qa_chain_or_error = load_rag()

def get_player(path):
    if not os.path.exists(path): return None
    with open(path, "rb") as f: b64 = base64.b64encode(f.read()).decode()
    return f'<audio autoplay controls src="data:audio/mp3;base64,{b64}" style="width:100%;"></audio>'

# ==================================================
# 🌟 核心修改：使用 fragment 進行局部更新
# ==================================================

st.title("🗺️ 雲科大隨身語音導覽")

# 初始化 session state
if 'user_pos' not in st.session_state:
    st.session_state.user_pos = None # 預設無位置

# 這一塊函式每 3 秒會自己跑一次，但「不會」讓整頁重新整理
@st.fragment(run_every=3) # 👈 這就是防閃爍的神奇指令 (需 Streamlit 1.37+)
def update_gps_loop():
    # 產生動態 ID
    gps_id = f"gps_{time.time()}"
    try:
        # 這裡只會更新這個隱藏的 GPS 元件，不會影響外面的地圖
        loc = get_geolocation(component_key=gps_id)
        if loc:
            lat = loc["coords"]["latitude"]
            lon = loc["coords"]["longitude"]
            
            # 只有當位置真的改變，且距離超過 5 公尺才更新全局變數 (減少無謂的重繪)
            old_pos = st.session_state.user_pos
            if old_pos:
                dist = geodesic(old_pos, (lat, lon)).meters
                if dist > 5: # 門檻：移動超過 5 公尺才更新地圖
                    st.session_state.user_pos = (lat, lon)
                    st.rerun() # 只有真的移動了，才觸發整頁刷新更新地圖
            else:
                # 第一次抓到位置
                st.session_state.user_pos = (lat, lon)
                st.rerun()
                
    except:
        pass
    
    # 顯示一個小小的狀態燈，證明它活著
    st.caption(f"📡 訊號偵測中... {int(time.time()) % 100}")

# 呼叫這個局部迴圈 (它會在背景一直跑)
update_gps_loop()

# ==================================================
# 下面是主畫面 (只有 st.session_state.user_pos 改變時才會重畫)
# ==================================================

if st.session_state.user_pos:
    user_pos = st.session_state.user_pos
    
    # 計算最近景點
    m = folium.Map(location=user_pos, zoom_start=17)
    folium.Marker(user_pos, popup="我", icon=folium.Icon(color="blue", icon="user")).add_to(m)
    
    nearest_key = None
    min_dist = float("inf")

    for key, info in SPOTS.items():
        spot_pos = (info["lat"], info["lon"])
        d = geodesic(user_pos, spot_pos).meters
        folium.Marker(spot_pos, popup=f"{info['name']} ({int(d)}m)", icon=folium.Icon(color="red", icon="info-sign")).add_to(m)
        folium.Circle(spot_pos, radius=TRIGGER_DIST, color="red", fill=True, fill_opacity=0.1).add_to(m)
        if d < min_dist:
            min_dist = d
            nearest_key = key

    st_folium(m, width=700, height=350)
    
    # 互動區
    if nearest_key and min_dist <= TRIGGER_DIST:
        spot = SPOTS[nearest_key]
        st.success(f"📍 抵達：**{spot['name']}**")
        
        lang = st.radio("語言", ["中文", "台語"], horizontal=True)
        intro_text = spot["intro_cn"] if lang == "中文" else spot.get("intro_tw", "無資料")
        st.info(intro_text)
        
        if st.button("▶ 播放"):
            suffix = "cn" if lang == "中文" else "tw"
            path = f"data/audio/{nearest_key}_{suffix}.mp3"
            if suffix == "tw" and not os.path.exists(path):
                path = f"data/audio/{nearest_key}_cn.mp3"
            player = get_player(path)
            if player: st.markdown(player, unsafe_allow_html=True)

        st.divider()
        user_q = st.chat_input("有什麼問題想問導覽員？")
        if user_q:
            if isinstance(qa_chain_or_error, str):
                st.error(qa_chain_or_error)
            else:
                with st.spinner("AI 思考中..."):
                    resp = qa_chain_or_error.invoke(f"地點:{spot['name']}, 問題:{user_q}")
                    st.write(resp)
    else:
        st.info(f"🚶 前往最近景點：{SPOTS[nearest_key]['name']} (還有 {int(min_dist - TRIGGER_DIST)}m)")

else:
    st.warning("📡 首次定位中... 請稍候")