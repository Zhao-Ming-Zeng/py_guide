import json
import os
import toml
import asyncio
import requests
import base64
import edge_tts
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 讀取設定 ---
try:
    secrets = toml.load(".streamlit/secrets.toml")
    os.environ["GOOGLE_API_KEY"] = secrets["GOOGLE_API_KEY"]
    YATING_KEY = secrets["YATING_API_KEY"]
except Exception as e:
    print(f"❌ 設定檔讀取失敗: {e}")
    exit()

# --- 函式定義 ---
def translate_to_tw(text):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    return llm.invoke(f"請將此中文改寫為台語口語漢字，直接輸出結果：{text}").content.strip()

async def gen_cn_mp3(text, path):
    print(f"   🎙️ 生成中文語音...")
    await edge_tts.Communicate(text, "zh-TW-HsiaoChenNeural").save(path)

def gen_tw_mp3(text, path):
    print(f"   🎙️ 生成台語語音 (雅婷)...")
    try:
        res = requests.post(
            "https://api.yating.tw/v2/text-to-speech/synthesize",
            json={
                "input": {"text": text, "type": "text"},
                "voice": {"model": "zh_en_female_1", "speed": 1.0, "pitch": 1.0},
                "audioConfig": {"encoding": "MP3", "sampleRate": 22050}
            },
            headers={"Authorization": f"Key {YATING_KEY}", "Content-Type": "application/json"}
        )
        if res.status_code == 201:
            with open(path, "wb") as f: f.write(base64.b64decode(res.json()["audioContent"]))
        else:
            print(f"❌ 雅婷 API 錯誤: {res.text}")
    except Exception as e:
        print(f"❌ 連線錯誤: {e}")

# --- 主程式 ---
async def main():
    if not os.path.exists('data/spots.json'):
        print("❌ 找不到 data/spots.json")
        return

    os.makedirs("data/audio", exist_ok=True)
    with open('data/spots.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("🚀 開始處理資源...")
    
    for key, info in data.items():
        print(f"\n📍 處理：{info['name']}")
        
        # 1. 翻譯台語
        if not info.get('intro_tw'):
            info['intro_tw'] = translate_to_tw(info['intro_cn'])
            print("   ✅ 台語翻譯完成")

        # 2. 生成音檔
        cn_path = f"data/audio/{key}_cn.mp3"
        tw_path = f"data/audio/{key}_tw.mp3"

        if not os.path.exists(cn_path): await gen_cn_mp3(info['intro_cn'], cn_path)
        if not os.path.exists(tw_path): gen_tw_mp3(info['intro_tw'], tw_path)

    # 存回 JSON
    with open('data/spots.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("\n🎉 全部完成！")

if __name__ == "__main__":
    asyncio.run(main())