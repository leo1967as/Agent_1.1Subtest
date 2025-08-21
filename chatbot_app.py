import streamlit as st
import os
import requests
import json
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional

# --- 1. โหลดค่า Config และ Environment Variables ---
try:
    import config
except ImportError:
    st.error("ไม่พบไฟล์ config.py กรุณาสร้างไฟล์ตามคำแนะนำ")
    st.stop()

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# --- 2. Prompt Template ตามที่กำหนด ---
RESEARCH_PROMPT_TEMPLATE = """
คำสั่ง: คุณคือผู้ช่วยทนายความ AI ระดับ Senior Associate ที่มีความเชี่ยวชาญในการวิเคราะห์ข้อเท็จจริงและสังเคราะห์ข้อมูลทางกฎหมาย ภารกิจของคุณคือการร่าง "บันทึกความเห็นทางกฎหมายเบื้องต้น" (Preliminary Legal Memorandum) โดยต้อง **ยึดตามข้อมูลที่ได้รับมาอย่างเคร่งครัดที่สุด**

**ข้อมูลนำเข้า:**
1.  **ข้อเท็จจริงในคดี (Case Facts):** {case_facts}
2.  **เอกสารอ้างอิง (Context):** {context} (ประกอบด้วยตัวบทกฎหมายและแนวคำพิพากษา/คำวินิจฉัยที่เกี่ยวข้อง)

**งานของคุณ:** จงวิเคราะห์ข้อเท็จจริงและสังเคราะห์ข้อมูลจาก **"เอกสารอ้างอิง (Context)" เท่านั้น** เพื่อร่างบันทึกความเห็นตามโครงสร้างข้างล่างนี้อย่างละเอียดและเป็นขั้นตอน:
---
### ๑. สรุปข้อเท็จจริง
(สรุปย่อข้อเท็จจริงในคดีที่ได้รับมา เพื่อยืนยันความเข้าใจ)

### ๒. ประเด็นข้อกฎหมายที่ต้องวินิจฉัย
(ระบุประเด็นทางกฎหมายที่เกิดขึ้นจากข้อเท็จจริง)

### ๓. ข้อกฎหมายที่เกี่ยวข้อง
(ระบุตัวบทกฎหมายที่ค้นเจอใน **"เอกสารอ้างอิง (Context)" เท่านั้น** พร้อมอธิบายหลักการสำคัญสั้นๆ **หากใน Context ไม่มีตัวบทกฎหมายโดยตรง ให้ระบุว่า 'ไม่พบตัวบทกฎหมายที่เกี่ยวข้องโดยตรงในเอกสารอ้างอิง'**)

### ๔. แนวคำพิพากษา/คำวินิจฉัยที่เทียบเคียง
(นำเสนอแนวคำพิพากษา/คำวินิจฉัยที่เกี่ยวข้องจาก **"เอกสารอ้างอิง (Context)" เท่านั้น** โดยสรุปข้อเท็จจริงในคดีนั้นๆ และหลักกฎหมายที่ศาลวางไว้ พร้อมอ้างอิงเลขที่คดีให้ชัดเจน)

### ๕. การปรับใช้กฎหมายกับข้อเท็จจริง
(นำข้อกฎหมายและแนวทางจากข้อ ๓ และ ๔ มาปรับใช้กับข้อเท็จจริงในคดีนี้อย่างละเอียด เพื่อวินิจฉัยในแต่ละประเด็น)

### ๖. สรุปและข้อเสนอแนะเบื้องต้น
(สรุปผลการวิเคราะห์ทั้งหมด และให้ข้อเสนอแนะเบื้องต้นตามแนวทางที่ปรากฏในเอกสารอ้างอิง)
---
**กฎเหล็ก:** ห้ามอ้างอิงข้อมูลกฎหมายหรือฎีกาใดๆ ที่ไม่ได้ปรากฏอยู่ใน **"เอกสารอ้างอิง (Context)"** โดยเด็ดขาด จงใช้ภาษาทางกฎหมายที่ถูกต้องและอ้างอิงเลขที่คดีและมาตรากฎหมายเท่าที่พบในข้อมูลที่ให้มาเท่านั้น
"""

# --- 3. คลาสสำหรับจัดการการเชื่อมต่อ Backend (เหมือนเดิม) ---

class VectorDBConnector:
    """จัดการการเชื่อมต่อและการค้นหาใน ChromaDB"""
    def __init__(self, db_path: str, collection_name: str, embedding_model: str):
        try:
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_collection(name=collection_name)
            
            @st.cache_resource
            def get_embedding_model(model_name):
                print(f"Loading embedding model: {model_name}")
                return SentenceTransformer(model_name, device='cuda' if torch.cuda.is_available() else 'cpu')
            
            self.model = get_embedding_model(embedding_model)
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ ChromaDB หรือโหลดโมเดล Embedding: {e}")
            st.stop()

    def search(self, query: str, n_results: int = 5) -> Optional[Dict]:
        """ค้นหาเอกสารที่เกี่ยวข้อง (เพิ่ม n_results เป็น 5 เพื่อให้บริบทครอบคลุมขึ้น)"""
        try:
            query_embedding = self.model.encode([query], normalize_embeddings=True)
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results
            )
            return results
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดขณะค้นหาใน Vector DB: {e}")
            return None

class LLMConnector:
    """จัดการการเชื่อมต่อและเรียกใช้ LLM ผ่าน OpenRouter API"""
    def __init__(self, api_key: str):
        if not api_key:
            st.error("ไม่ได้กำหนด OPENROUTER_API_KEY ในไฟล์ .env")
            st.stop()
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": config.YOUR_SITE_URL,
            "X-Title": config.YOUR_APP_NAME,
            "Content-Type": "application/json"
        }

    def get_completion(self, prompt: str) -> Optional[str]:
        """ส่ง prompt ไปยัง LLM และรับคำตอบกลับมา"""
        payload = {
            "model": config.LLM_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 4096 # เพิ่ม max_tokens เพื่อรองรับเอกสารที่ยาวขึ้น
        }
        try:
            response = requests.post(config.OPENROUTER_API_URL, headers=self.headers, json=payload, timeout=120) # เพิ่ม timeout
            response.raise_for_status()
            result = response.json()
            return result.get('choices', [{}])[0].get('message', {}).get('content')
        except requests.exceptions.RequestException as e:
            st.error(f"การเชื่อมต่อ API ล้มเหลว: {e}")
            return None
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            st.error(f"ไม่สามารถอ่านผลลัพธ์จาก API ได้: {e}")
            return None

class LegalMemoGenerator:
    """Orchestrator ที่รวมการทำงานของ VectorDB และ LLM เพื่อสร้างบันทึกความเห็น"""
    def __init__(self, vector_db: VectorDBConnector, llm: LLMConnector):
        self.vector_db = vector_db
        self.llm = llm

    def generate_memo(self, case_facts: str) -> Tuple[Optional[str], List[str]]:
        """สร้างบันทึกความเห็นทางกฎหมายจากข้อเท็จจริง"""
        # 1. ค้นหาข้อมูล (Retrieve)
        search_results = self.vector_db.search(case_facts)
        
        if not search_results or not search_results.get('documents') or not search_results['documents'][0]:
            st.warning("ไม่พบข้อมูลอ้างอิงที่เกี่ยวข้องกับข้อเท็จจริงที่ให้มา")
            return None, []

        # 2. สร้าง Context (Augment)
        context_docs = search_results['documents'][0]
        metadatas = search_results['metadatas'][0]
        
        context_str = ""
        for i, (doc, meta) in enumerate(zip(context_docs, metadatas)):
            case_num = meta.get('case_number', 'N/A')
            context_str += f"--- เอกสารอ้างอิง {i+1} (เลขคดี: {case_num}) ---\n"
            context_str += f"{doc}\n\n"

        # 3. สร้าง Prompt ที่สมบูรณ์
        final_prompt = RESEARCH_PROMPT_TEMPLATE.format(context=context_str, case_facts=case_facts)
        
        # 4. สร้างคำตอบ (Generate)
        response = self.llm.get_completion(final_prompt)
        
        sources = [f"**เลขคดี {meta.get('case_number', 'N/A')}:**\n{doc[:250]}..." for doc, meta in zip(context_docs, metadatas)]
        return response, sources

# --- 4. ส่วนของ Streamlit User Interface ---

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="เครื่องมือร่างบันทึกความเห็นทางกฎหมาย", layout="wide")
st.title("📑 เครื่องมือช่วยร่างบันทึกความเห็นทางกฎหมายเบื้องต้น")
st.caption("ขับเคลื่อนโดย Llama 3.1 70B และฐานข้อมูลคำวินิจฉัย (RAG)")

# ใช้ @st.cache_resource เพื่อให้สร้าง object แค่ครั้งเดียว
@st.cache_resource
def get_memo_generator():
    """สร้างและคืนค่า LegalMemoGenerator instance"""
    vector_db = VectorDBConnector(
        db_path=config.DB_PATH, 
        collection_name=config.COLLECTION_NAME, 
        embedding_model=config.EMBEDDING_MODEL_NAME
    )
    llm = LLMConnector(api_key=OPENROUTER_API_KEY)
    return LegalMemoGenerator(vector_db, llm)

# ตรวจสอบ API Key ก่อนเริ่ม
if not OPENROUTER_API_KEY:
    st.warning("กรุณาตั้งค่า OPENROUTER_API_KEY ในไฟล์ .env ก่อนเริ่มใช้งาน")
else:
    memo_generator = get_memo_generator()
    
    st.info("กรุณาป้อนข้อเท็จจริงของคดีที่ต้องการวิเคราะห์ในช่องด้านล่าง แล้วกดปุ่มเพื่อเริ่ม")

    # รับ Input จากผู้ใช้
    case_facts_input = st.text_area("กรุณาป้อนข้อเท็จจริงในคดี:", height=250, placeholder="ตัวอย่าง: นายดำต้องการฟ้องนายขาวเรื่องผิดสัญญาซื้อขายที่ดิน โดยนายขาวอ้างว่าสัญญาเป็นโมฆะเพราะไม่ได้ทำเป็นหนังสือและจดทะเบียนต่อพนักงานเจ้าหน้าที่...")

    # ปุ่มสำหรับเริ่มทำงาน
    if st.button("🚀 เริ่มการวิเคราะห์และร่างบันทึกความเห็น", type="primary"):
        if not case_facts_input.strip():
            st.error("กรุณาป้อนข้อเท็จจริงในคดีก่อน")
        else:
            with st.spinner("กำลังค้นหาข้อมูล, วิเคราะห์, และร่างบันทึกความเห็น... (ขั้นตอนนี้อาจใช้เวลา 1-2 นาที)"):
                memo, sources = memo_generator.generate_memo(case_facts_input)

            st.divider()

            if memo:
                st.subheader("📝 บันทึกความเห็นทางกฎหมายเบื้องต้น (ฉบับร่าง)")
                st.markdown(memo)

                # แสดงแหล่งข้อมูลอ้างอิงใน expander
                if sources:
                    with st.expander("ดูเอกสารอ้างอิงที่ใช้ในการร่าง"):
                        for source in sources:
                            st.success(source)
            else:
                st.error("ขออภัยครับ ไม่สามารถสร้างบันทึกความเห็นได้ในขณะนี้ กรุณาลองใหม่อีกครั้ง")