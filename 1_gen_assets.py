import json
import os
import toml
import asyncio
import edge_tts
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from yating_tts_sdk import YatingClient
except ImportError:
    print("❌ 尚未安裝 SDK！請執行: pip install yating-tts-sdk")
    exit()

# --- 讀取設定 ---
try:
    secrets = toml.load(".streamlit/secrets.toml")
    GOOGLE_KEY = secrets["GOOGLE_API_KEY"]
    YATING_KEY = secrets["YATING_API_KEY"]
except Exception as e:
    print(f"❌ 設定檔讀取失敗: {e}")
    exit()

# --- 函式定義 ---
def translate_to_tw(text):
    try:
        # 如果更新後還是找不到 1.5-flash，這裡會自動降級用 gemini-pro
        model_name = "gemini-1.5-flash"
        
        llm = ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=0.7,
            google_api_key=GOOGLE_KEY
        )
        return llm.invoke(f"請將此中文改寫為台語口語漢字，直接輸出結果：{text}").content.strip()
    except Exception as e:
        print(f"   ⚠️ 翻譯失敗 (原因: {e})")
        print("   💡 嘗試降級使用 'gemini-pro'...")
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, google_api_key=GOOGLE_KEY)
            return llm.invoke(f"請將此中文改寫為台語口語漢字，直接輸出結果：{text}").content.strip()
        except:
            return text # 真的不行就回傳原文

async def gen_cn_mp3(text, path):
    print(f"   🎙️ 生成中文語音 (Edge TTS)...")
    communicate = edge_tts.Communicate(text, "zh-TW-HsiaoChenNeural")
    await communicate.save(path)

# ⭐️ 核心修正：SDK 改用 V1 網址
def gen_tw_mp3_sdk(text, path):
    print(f"   🎙️ 嘗試生成台語語音 (雅婷 SDK V1)...")
    
    # ✅ 改用 V1 網址 (最穩定)
    url = "https://api.yating.tw/v1/text-to-speech/synthesize"
    
    try:
        client = YatingClient(url, YATING_KEY)
        
        client.synthesize(
            text,               # text
            "text",             # type
            "zh_en_female_1",   # model
            1.0,                # speed
            1.0,                # pitch
            1.0,                # energy
            "MP3",              # encoding
            "22K",              # sample_rate (SDK 會自動處理字串/數字轉換)
            path                # file_name
        )
        
        print("      ✅ 雅婷 SDK 生成成功！")
        return True
        
    except Exception as e:
        print(f"      ⚠️ SDK 執行失敗: {e}")
        return False

# --- 主程式 ---
async def main():
    if not os.path.exists('data/spots.json'):
        print("❌ 找不到 data/spots.json")
        return

    os.makedirs("data/audio", exist_ok=True)
    with open('data/spots.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("🚀 開始處理資源 (V1 SDK 版)...")
    
    for key, info in data.items():
        print(f"\n📍 處理：{info['name']}")
        
        # 1. 翻譯
        if not info.get('intro_tw'):
            info['intro_tw'] = translate_to_tw(info['intro_cn'])

        # 2. 生成中文
        cn_path = f"data/audio/{key}_cn.mp3"
        if not os.path.exists(cn_path):
            await gen_cn_mp3(info['intro_cn'], cn_path)
        
        # 3. 生成台語
        tw_path = f"data/audio/{key}_tw.mp3"
        
        # 刪除舊檔
        if os.path.exists(tw_path):
            os.remove(tw_path)
            
        # 呼叫 SDK
        success = gen_tw_mp3_sdk(info['intro_tw'], tw_path)
        
        if not success:
            print("      ⚠️ 生成失敗 (請確認 Key 有 V1/V2 權限)")

    with open('data/spots.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("\n🎉 全部處理完成！")

if __name__ == "__main__":
    asyncio.run(main())