import streamlit as st
import json
import os
import base64
import folium
from streamlit_folium import st_folium
from streamlit_js_eval import get_geolocation
# ❌ 移除 st_autorefresh，我們改用更高級的 fragment
from geopy.distance import geodesic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- 1. 設定頁面 ---
st.set_page_config(page_title="語音導覽", layout="wide", page_icon="🗺️")

# --- 2. CSS 樣式 (隱藏不需要的元素) ---
st.markdown("""
<style>
    /* 美化播放按鈕 */
    .stButton button {
        background-color: #E63946; color: white; border-radius: 50px;
        font-size: 20px; border: 2px solid white;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.3);
        width: 100%;
    }
    .stButton button:hover { background-color: #D62828; transform: scale(1.02); }
</style>
""", unsafe_allow_html=True)

# --- 3. 載入資料 ---
if not os.path.exists("data/spots.json"):
    st.error("❌ 找不到 data/spots.json")
    st.stop()
SPOTS = json.load(open("data/spots.json", "r", encoding="utf-8"))
TRIGGER_DIST = 150

# --- 4. 初始化 Session State (全域變數) ---
if 'current_spot' not in st.session_state:
    st.session_state.current_spot = None # 目前所在的景點 ID
if 'user_coords' not in st.session_state:
    st.session_state.user_coords = None # 使用者座標

# --- 5. RAG 模型 ---
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

# --- 6. 隱形播放器函式 (解決進度條問題) ---
def play_audio_hidden(path):
    if not os.path.exists(path):
        st.toast("⚠️ 找不到音檔", icon="❌")
        return
    
    with open(path, "rb") as f:
        audio_bytes = f.read()
    b64 = base64.b64encode(audio_bytes).decode()
    
    # 使用 HTML5 audio 標籤，設定 autoplay 且不顯示 controls (hidden)
    # 這樣就完全看不到進度條，只有聲音
    sound_html = f"""
    <audio autoplay style="display:none">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    <script>
        // 強制嘗試播放 (針對部分瀏覽器限制)
        var audio = document.querySelector("audio");
        audio.play().catch(function(error) {{
            console.log("Autoplay blocked: " + error);
        }});
    </script>
    """
    # 使用一個空的 container 注入 HTML，這樣不會佔版面
    st.markdown(sound_html, unsafe_allow_html=True)
    st.toast("▶️ 開始播放導覽", icon="🎧")

# ==========================================================
# 🌟 核心技術：GPS 與地圖的「局部刷新」 (Fragment)
# 只有這個函式會每 3 秒重跑，其他的程式碼都靜止不動！
# ==========================================================
@st.fragment(run_every=3)
def map_gps_tracker():
    # 1. 取得 GPS (使用動態 Key 強制更新)
    import time
    gps_id = f"gps_{int(time.time())}"
    
    try:
        loc = get_geolocation(component_key=gps_id)
    except:
        loc = None

    # 2. 處理座標
    current_pos = st.session_state.user_coords # 預設用舊的
    
    if loc:
        lat = loc["coords"]["latitude"]
        lon = loc["coords"]["longitude"]
        current_pos = (lat, lon)
        st.session_state.user_coords = current_pos # 更新全域變數
    elif st.session_state.user_coords is None:
        # 如果完全沒座標，給預設值 (雲科大)
        current_pos = (23.694, 120.534)

    # 3. 畫地圖
    m = folium.Map(location=current_pos, zoom_start=17)
    if loc:
        folium.Marker(current_pos, popup="我", icon=folium.Icon(color="blue", icon="user")).add_to(m)

    # 4. 計算距離與最近景點
    nearest_key = None
    min_dist = float("inf")
    
    for key, info in SPOTS.items():
        spot_pos = (info["lat"], info["lon"])
        d = geodesic(current_pos, spot_pos).meters
        
        folium.Marker(spot_pos, popup=f"{info['name']} ({int(d)}m)", icon=folium.Icon(color="red", icon="info-sign")).add_to(m)
        folium.Circle(spot_pos, radius=TRIGGER_DIST, color="red", fill=True, fill_opacity=0.1).add_to(m)
        
        if d < min_dist:
            min_dist = d
            nearest_key = key

    # 5. 顯示地圖 (在這個 fragment 裡顯示)
    st_folium(m, width=700, height=300)
    
    # 6. 【關鍵】判斷是否切換景點
    # 如果我們進入了新的景點，或者離開了景點，這時候才需要通知外面的「聊天室」更新
    # 這樣可以避免聊天室每 3 秒閃一次
    
    new_spot = None
    if nearest_key and min_dist <= TRIGGER_DIST:
        new_spot = nearest_key
        st.success(f"📍 抵達：{SPOTS[new_spot]['name']}")
    else:
        if nearest_key:
            st.info(f"🚶 前往：{SPOTS[nearest_key]['name']} (還有 {int(min_dist - TRIGGER_DIST)}m)")

    # 只有當「景點改變」時，才觸發全域刷新 (Rerun)
    # 這樣平常打字時就不會被干擾，只有走到下一個景點時才會刷新一次
    if new_spot != st.session_state.current_spot:
        st.session_state.current_spot = new_spot
        st.rerun()

# ==========================================================
# 主程式 (這裡是靜止的，不會一直閃)
# ==========================================================
st.title("🗺️ 雲科大隨身語音導覽")

col1, col2 = st.columns([3, 2])

with col1:
    # 呼叫那個會自己動的地圖 Fragment
    map_gps_tracker()

with col2:
    # 這裡的介面是穩定的，不會因為 GPS 更新而被重置
    
    current_spot_key = st.session_state.current_spot
    
    if current_spot_key:
        spot_info = SPOTS[current_spot_key]
        st.subheader(f"🏛️ {spot_info['name']}")
        
        # 語言選擇
        lang = st.radio("導覽語言", ["中文", "台語"], horizontal=True)
        intro_text = spot_info["intro_cn"] if lang == "中文" else spot_info.get("intro_tw", "無資料")
        
        # 文字介紹框 (可捲動)
        st.text_area("介紹", intro_text, height=150)
        
        # ▶️ 播放按鈕 (完全無進度條版)
        if st.button("▶️ 點擊播放語音導覽"):
            suffix = "cn" if lang == "中文" else "tw"
            path = f"data/audio/{current_spot_key}_{suffix}.mp3"
            if suffix == "tw" and not os.path.exists(path):
                path = f"data/audio/{current_spot_key}_cn.mp3"
            
            # 呼叫隱形播放函式
            play_audio_hidden(path)

        st.divider()
        
        # 💬 AI 聊天室 (因為在 Main 區域，所以不會被 GPS 刷新打斷)
        st.markdown("### 🤖 導覽小幫手")
        user_q = st.chat_input("對這裡有什麼好奇嗎？問我吧！")
        
        if user_q:
            if isinstance(qa_chain_or_error, str):
                st.error(qa_chain_or_error)
            else:
                with st.chat_message("user"):
                    st.write(user_q)
                with st.chat_message("assistant"):
                    with st.spinner("思考中..."):
                        # 將景點資訊帶入 Prompt
                        full_q = f"我現在在「{spot_info['name']}」，{user_q}"
                        resp = qa_chain_or_error.invoke(full_q)
                        st.write(resp)
    else:
        st.markdown("""
        ### 👋 歡迎使用
        請移動您的腳步，地圖左側會顯示您的位置。
        當您進入景點範圍 (紅圈) 時，這裡會自動出現導覽資訊與 AI 問答功能。
        """)