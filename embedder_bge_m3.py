# embedder_bge_m3.py
import os
import json
import logging
import time
import torch
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, ValidationError
from tqdm import tqdm

# --- 1. ตั้งค่าระบบบันทึก Log และ Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# --- Configuration ---
MODEL_NAME = "BAAI/bge-m3"
INPUT_JSON_PATH = "processed_cases/_ALL_CASES_COMBINED.json"
OUTPUT_PICKLE_PATH = "embeddings_bge_m3.pkl" # .pkl เหมาะสำหรับเก็บ NumPy arrays
BATCH_SIZE = 128 # ปรับค่านี้ได้ตามขนาด VRAM ของ GPU (ค่าสูงขึ้นใช้ VRAM มากขึ้น แต่เร็วขึ้น)

# --- 2. Pydantic Models สำหรับตรวจสอบข้อมูล Input ---
# ใช้โครงสร้างเดียวกับสคริปต์ก่อนหน้าเพื่อความเข้ากันได้
class Court(BaseModel):
    name: str
    role: str

class Party(BaseModel):
    name: str
    role: str

class LegalCase(BaseModel):
    document_type: str
    case_number: str
    involved_courts: List[Court]
    parties: List[Party]
    referenced_laws: List[str]
    case_background_full: str | None
    plaintiffs_argument_full: str | None
    defendants_argument_full: str | None
    committee_reasoning_full: str | None
    final_decision: str | None

# --- 3. ฟังก์ชันสำหรับเตรียมข้อมูล ---
def load_and_validate_cases(filepath: str) -> List[LegalCase]:
    """โหลดข้อมูลจากไฟล์ JSON และตรวจสอบความถูกต้องด้วย Pydantic"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # --- START OF THE PROPOSED FIX ---
        # ตรวจสอบว่าข้อมูลเป็น List หรือไม่
        if not isinstance(data, list):
            logging.error(f"ข้อมูลในไฟล์ '{filepath}' ไม่ได้อยู่ในรูปแบบ List ของคดี")
            return []

        case_list = []
        # ตรวจสอบว่าข้อมูลเป็น List ซ้อน List หรือไม่ และทำการ Flatten
        if len(data) > 0 and isinstance(data[0], list):
            logging.warning(f"ตรวจพบโครงสร้างข้อมูลแบบ List ซ้อน List ในไฟล์ '{filepath}'. กำลังทำการรวมข้อมูล (Flattening)...")
            # ใช้ list comprehension เพื่อคลี่ list ที่ซ้อนกันออกมาเป็น list เดียว
            case_list = [case for sublist in data for case in sublist]
        else:
            # ข้อมูลเป็น List ของ Dictionary อยู่แล้ว
            case_list = data
        # --- END OF THE PROPOSED FIX ---

        validated_cases = [LegalCase.model_validate(case) for case in case_list]
        logging.info(f"โหลดและตรวจสอบข้อมูลสำเร็จ พบทั้งหมด {len(validated_cases)} คดี")
        return validated_cases
    except FileNotFoundError:
        logging.error(f"ไม่พบไฟล์ Input: '{filepath}' กรุณารันสคริปต์ประมวลผลก่อน")
        return []
    except ValidationError as e:
        logging.error(f"ข้อมูลในไฟล์ JSON ไม่ตรงตามโครงสร้างที่กำหนด: {e}")
        return []
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดที่ไม่คาดคิดขณะโหลดไฟล์: {e}")
        return []
    
def prepare_texts_for_embedding(cases: List[LegalCase]) -> tuple[List[str], List[str]]:
    """
    รวมเนื้อหาทั้งหมดของแต่ละคดีให้เป็นข้อความเดียวที่สมบูรณ์เพื่อนำไป Embedding
    """
    all_texts = []
    all_case_numbers = []
    for case in cases:
        # รวมเนื้อหาทั้งหมดเข้าด้วยกัน โดยมีหัวข้อกำกับเพื่อรักษาบริบท
        full_text = (
            f"เลขคดี: {case.case_number}\n\n"
            f"ความเป็นมาของคดี:\n{case.case_background_full or 'ไม่ปรากฏข้อมูล'}\n\n"
            f"ข้อกล่าวอ้างของโจทก์:\n{case.plaintiffs_argument_full or 'ไม่ปรากฏข้อมูล'}\n\n"
            f"ข้อต่อสู้ของจำเลย:\n{case.defendants_argument_full or 'ไม่ปรากฏข้อมูล'}\n\n"
            f"เหตุผลของคณะกรรมการ:\n{case.committee_reasoning_full or 'ไม่ปรากฏข้อมูล'}\n\n"
            f"ผลคำวินิจฉัย:\n{case.final_decision or 'ไม่ปรากฏข้อมูล'}"
        )
        all_texts.append(full_text)
        all_case_numbers.append(case.case_number)
    logging.info(f"เตรียมข้อความสำหรับ Embedding ทั้งหมด {len(all_texts)} ชุดเรียบร้อยแล้ว")
    return all_texts, all_case_numbers

# --- 4. ส่วนหลักของการทำงาน ---
def main():
    # --- ตรวจสอบ CUDA ---
    if not torch.cuda.is_available():
        logging.error("ไม่พบ CUDA device! สคริปต์นี้ต้องการ GPU ของ NVIDIA เพื่อทำงาน")
        return

    logging.info(f"พบ CUDA device: {torch.cuda.get_device_name(0)}")

    # --- 1. โหลดและเตรียมข้อมูล ---
    legal_cases = load_and_validate_cases(INPUT_JSON_PATH)
    if not legal_cases:
        logging.error("ไม่สามารถดำเนินงานต่อได้เนื่องจากไม่มีข้อมูล")
        return
    
    texts_to_embed, case_numbers = prepare_texts_for_embedding(legal_cases)

    # --- 2. โหลดโมเดล bge-m3 พร้อมการตั้งค่าประสิทธิภาพสูงสุด ---
    logging.info(f"กำลังโหลดโมเดล '{MODEL_NAME}' ลงบน CUDA...")
    
    # *** คำอธิบายเชิงลึก (ส่วนที่ทำให้เร็ว) ***
    # 1. device='cuda': บังคับให้โมเดลทำงานบน GPU
    # 2. torch_dtype=torch.float16: ใช้ 16-bit precision ลดการใช้ VRAM และเร่งความเร็วบน Tensor Cores
    # 3. trust_remote_code=True: จำเป็นสำหรับโมเดลบางตัว รวมถึง bge-m3
    # 4. Flash Attention 2: หากคุณติดตั้ง `flash-attn` ไลบรารี `sentence-transformers`
    #    จะตรวจจับและใช้งานให้โดยอัตโนมัติ ซึ่งจะเร่งความเร็วขึ้นไปอีก 2-4 เท่า!
    try:
        model = SentenceTransformer(
            MODEL_NAME,
            device='cuda',
            # torch_dtype=torch.float16,
            trust_remote_code=True
        )
        logging.info("โหลดโมเดลสำเร็จ!")
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดขณะโหลดโมเดล: {e}")
        logging.error("อาจเกิดจากปัญหาการเชื่อมต่ออินเทอร์เน็ต หรือไฟล์โมเดลใน Cache เสียหาย")
        return

    # --- 3. สร้าง Embeddings ---
    logging.info(f"เริ่มสร้าง Embeddings สำหรับ {len(texts_to_embed)} เอกสาร (Batch Size: {BATCH_SIZE})...")
    start_time = time.time()

    embeddings = model.encode(
        texts_to_embed,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True # แนะนำให้ทำเสมอสำหรับ bge models
    )

    end_time = time.time()
    processing_time = end_time - start_time
    docs_per_second = len(texts_to_embed) / processing_time

    logging.info("สร้าง Embeddings ทั้งหมดเรียบร้อยแล้ว!")
    logging.info(f"ขนาดของ Embeddings: {embeddings.shape}") # ควรเป็น (จำนวนเอกสาร, 1024) สำหรับ bge-m3
    logging.info(f"ใช้เวลาทั้งหมด: {processing_time:.2f} วินาที ({docs_per_second:.2f} เอกสาร/วินาที)")

    # --- 4. บันทึกผลลัพธ์ ---
    logging.info(f"กำลังบันทึกผลลัพธ์ลงในไฟล์ '{OUTPUT_PICKLE_PATH}'...")
    
    # จัดเก็บในรูปแบบ Dictionary เพื่อให้ง่ายต่อการนำไปใช้
    output_data = {
        "case_numbers": case_numbers,
        "texts": texts_to_embed,
        "embeddings": embeddings
    }

    import pickle
    with open(OUTPUT_PICKLE_PATH, "wb") as f_out:
        pickle.dump(output_data, f_out)

    logging.info("🎉 บันทึกไฟล์สำเร็จ! กระบวนการทั้งหมดเสร็จสมบูรณ์")

if __name__ == "__main__":
    main()