# config.py

# URL ของ OpenRouter API
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ชื่อโมเดลที่ต้องการใช้บน OpenRouter
# ตรวจสอบชื่อล่าสุดได้ที่: https://openrouter.ai/models
MODEL_NAME = "meta-llama/llama-3.3-70b-instruct"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
# ข้อมูลสำหรับ Header (ตามคำแนะนำของ OpenRouter)
YOUR_SITE_URL = "http://localhost:8000"
YOUR_APP_NAME = "Legal Document Processor v2"

# เวลารอสูงสุดสำหรับการตอบกลับจาก API (วินาที)
API_TIMEOUT = 180

DB_PATH = "legal_vector_db"
COLLECTION_NAME = "legal_cases_bge_m3"  # <--- ตรวจสอบบรรทัดนี้ให้ดี
LLM_MODEL_NAME = "meta-llama/llama-3.1-70b-instruct"  # <--- ตรวจสอบบรรทัดนี้ให้ดี
