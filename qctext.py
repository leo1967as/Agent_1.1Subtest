import os
import requests
import json
import sys
import re
import logging
import time
import argparse
from typing import Optional, Dict, List
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, Field
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- 1. ตั้งค่าระบบบันทึก Log ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# --- 2. โหลดค่าตั้งค่าจากไฟล์ ---
try:
    import config
except ImportError:
    logging.error("ไม่พบไฟล์ config.py กรุณาสร้างไฟล์ตามคำแนะนำ")
    sys.exit(1)

# --- 3. Pydantic Models ---
class Court(BaseModel):
    name: str
    role: str

class Party(BaseModel):
    name: str
    role: str

class LegalCase(BaseModel):
    document_type: str
    # Pydantic Model ต้องเป็น str เพราะโค้ดของเรารับประกันว่าจะมีค่านี้ก่อนบันทึกเสมอ
    case_number: str 
    involved_courts: List[Court]
    parties: List[Party]
    referenced_laws: List[str]
    case_background_full: Optional[str] = Field(default=None)
    plaintiffs_argument_full: Optional[str] = Field(default=None)
    defendants_argument_full: Optional[str] = Field(default=None)
    committee_reasoning_full: Optional[str] = Field(default=None)
    final_decision: Optional[str] = Field(default=None)

# --- 4. คลาสสำหรับเชื่อมต่อ API (มี Retry Logic) ---
class OpenRouterConnector:
    def __init__(self, api_key: str):
        if not api_key: raise ValueError("!!! ข้อผิดพลาดร้ายแรง: ไม่ได้กำหนด OPENROUTER_API_KEY ในไฟล์ .env")
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {self.api_key}", "HTTP-Referer": config.YOUR_SITE_URL, "X-Title": config.YOUR_APP_NAME, "Content-Type": "application/json"}
        self.session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get_completion(self, prompt: str) -> Optional[str]:
        timeout = getattr(config, 'API_TIMEOUT', 120)
        model_name = getattr(config, 'MODEL_NAME', 'meta-llama/llama-3.1-70b-instruct')
        payload = {
            "model": model_name, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "top_p": 1.0, "max_tokens": 8192
        }
        try:
            response = self.session.post(url=config.OPENROUTER_API_URL, headers=self.headers, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            return result.get('choices', [{}])[0].get('message', {}).get('content')
        except requests.exceptions.RequestException as e:
            logging.error(f"การเชื่อมต่อ API ล้มเหลวถาวรหลังจากลองใหม่หลายครั้ง: {e}")
            return None
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logging.error(f"เกิดข้อผิดพลาดในการอ่านผลลัพธ์: ไม่สามารถอ่านข้อมูล JSON จาก API ได้: {e}")
            if 'response' in locals(): logging.error(f"--- ข้อมูลดิบที่ได้รับ ---\n{response.text}\n--------------------")
            return None

# --- 5. คลาสหลักสำหรับประมวลผลเอกสาร ---
class LegalCaseProcessor:
    def __init__(self, connector: OpenRouterConnector):
        self.connector = connector

    def _preprocess_text(self, text: str) -> str:
        processed_text = re.sub(r'---\s*Page\s*\d+\s*---', '', text)
        processed_text = re.sub(r'\s{2,}', ' ', processed_text).strip()
        return processed_text

    def extract_data_for_single_case(self, single_case_text: str) -> Optional[dict]:
        clean_text = self._preprocess_text(single_case_text)
        prompt = f"""
[SYSTEM ROLE]
คุณคือ Automated Data Extraction Engine ที่มีความแม่นยำสูง ถูกโปรแกรมมาเพื่อแปลงเอกสารกฎหมายไทยให้เป็น Structured JSON เท่านั้น คุณไม่มีความคิดสร้างสรรค์ คุณทำงานตามกฎที่กำหนดอย่างเคร่งครัด 100%

[MISSION]
ภารกิจของคุณคือการทำตามขั้นตอนต่อไปนี้อย่างละเอียด:
1.  **Analyze**: วิเคราะห์ "Raw Text Input" ที่ได้รับมา
2.  **Extract**: สกัดข้อมูลตาม "Schema Definition & Extraction Logic" ที่กำหนดไว้อย่างแม่นยำ
3.  **Format**: สร้าง JSON Object ที่สมบูรณ์ตาม "Critical Output Rules"
4.  **Validate**: ตรวจสอบผลลัพธ์สุดท้ายกับ "Final Validation Checklist" ก่อนส่งมอบ

---
[RAW TEXT INPUT]
\"\"\"
{clean_text}
\"\"\"

---
[SCHEMA DEFINITION & EXTRACTION LOGIC]
คุณต้องสร้าง JSON Object ที่มีโครงสร้างตามนี้เท่านั้น โดยใช้ Logic การสกัดสำหรับแต่ละฟิลด์ดังนี้:

- **"document_type"**: (string) สกัดประเภทเอกสารจากหัวเรื่อง (เช่น "คำพิพากษาศาลฎีกา", "คำวินิจฉัยชี้ขาดอำนาจหน้าที่ระหว่างศาล")
- **"case_number"**: (string | null) สกัดหมายเลขคดีที่ชัดเจนที่สุดที่พบในเอกสาร (เช่น "คดีหมายเลขแดงที่ อ.1234/2567" หรือ "99/2560") **รักษาอักขระนำหน้าทั้งหมดไว้** หากไม่พบหมายเลขคดีใดๆ เลย ให้ใส่ค่าเป็น `null`
- **"involved_courts"**: (array of objects) สกัดรายชื่อศาลทั้งหมดที่ถูกกล่าวถึง แต่ละ object ต้องมี `{{ "name": "...", "role": "..." }}` หากไม่พบให้ใส่ `[]`
- **"parties"**: (array of objects) สกัดรายชื่อคู่กรณีทั้งหมด (โจทก์, จำเลย, ผู้ร้อง, ฯลฯ) แต่ละ object ต้องมี `{{ "name": "...", "role": "..." }}` หากไม่พบให้ใส่ `[]`
- **"referenced_laws"**: (array of strings) สกัดรายชื่อ "พระราชบัญญัติ", "ประมวลกฎหมาย", "รัฐธรรมนูญ" ที่ถูกอ้างอิงถึง หากไม่พบให้ใส่ `[]`
- **"case_background_full"**: (string | null) **[COPY-PASTE]** คัดลอกเนื้อหาทั้งหมดในส่วน "ความเป็นมาของคดี" ซึ่งมักจะเป็นย่อหน้าแรกๆ จนถึงก่อนส่วนของ "โจทก์ฟ้องว่า..."
- **"plaintiffs_argument_full"**: (string | null) **[COPY-PASTE]** คัดลอกเนื้อหาทั้งหมดในส่วนคำฟ้องของโจทก์ โดยมองหา Keyword เช่น "โจทก์ฟ้องว่า", "โจทก์กล่าวอ้างว่า"
- **"defendants_argument_full"**: (string | null) **[COPY-PASTE]** คัดลอกเนื้อหาทั้งหมดในส่วนคำให้การของจำเลย โดยมองหา Keyword เช่น "จำเลยให้การว่า"
- **"committee_reasoning_full"**: (string | null) **[COPY-PASTE]** คัดลอกเนื้อหาทั้งหมดในส่วนการให้เหตุผลทางกฎหมาย โดยมองหา Keyword เช่น "พิเคราะห์แล้ว", "ศาลเห็นว่า", "คณะกรรมการวินิจฉัยว่า" ไปจนถึงก่อนย่อหน้าสุดท้าย
- **"final_decision"**: (string | null) **[COPY-PASTE]** คัดลอกเนื้อหาทั้งหมดของย่อหน้าสุดท้ายที่เป็นคำพิพากษา โดยมองหา Keyword เช่น "พิพากษาว่า", "จึงมีคำวินิจฉัย"

---
[CRITICAL OUTPUT RULES]
1.  **VALID JSON ONLY**: ผลลัพธ์ต้องเป็น JSON Object ที่ถูกต้องตาม cú pháp (syntax) เท่านั้น ห้ามมีข้อความใดๆ นำหน้าหรือต่อท้ายบล็อก `{{...}}` โดยเด็ดขาด
2.  **ESCAPE DOUBLE QUOTES**: กฎที่สำคัญที่สุด: หากเนื้อหาที่คัดลอกมา (เช่น ในฟิลด์ `..._full`) มีเครื่องหมายคำพูด (`"`) อยู่ภายใน คุณ **ต้อง** ใส่ backslash (`\\`) ไว้ข้างหน้าเสมอ (ต้องเป็น `\\"`) เพื่อป้องกันไม่ให้โครงสร้าง JSON เสียหาย
3.  **HANDLE MISSING DATA**: หากไม่พบข้อมูลสำหรับฟิลด์ใด ให้ใส่ค่าเป็น `null` (สำหรับ string) หรือ `[]` (สำหรับ array) อย่าเว้นว่างหรือละทิ้งฟิลด์นั้นไป
4.  **NO SUMMARIZATION**: ห้ามสรุป, ย่อความ, หรือเปลี่ยนแปลงเนื้อหาในฟิลด์ที่ลงท้ายด้วย `_full` โดยเด็ดขาด ให้คัดลอกมาแบบคำต่อคำ

---
[FINAL VALIDATION CHECKLIST]
ก่อนจะให้คำตอบสุดท้าย จงตรวจสอบผลงานของคุณตาม Checklist นี้:
- [ ] ผลลัพธ์ทั้งหมดอยู่ในบล็อก `{{...}}` เดียวหรือไม่?
- [ ] เครื่องหมาย `"` ทั้งหมดภายใน value ของ string ถูก escape เป็น `\\"` แล้วใช่หรือไม่?
- [ ] ทุกฟิลด์ใน Schema ถูกสร้างขึ้นครบถ้วนหรือไม่? (ไม่มีฟิลด์ไหนหายไป)
- [ ] ฟิลด์ที่ไม่มีข้อมูล ถูกใส่ค่าเป็น `null` หรือ `[]` อย่างถูกต้องใช่หรือไม่?

เมื่อตรวจสอบครบถ้วนแล้ว จงส่งมอบ JSON Object ที่สมบูรณ์
"""
        llm_response_str = self.connector.get_completion(prompt)
        if not llm_response_str: return None
        try:
            json_match = re.search(r'\{.*\}', llm_response_str, re.DOTALL)
            if not json_match:
                logging.error(f"AI ไม่ได้ตอบกลับมาเป็น JSON (ไม่พบ JSON block)\n--- คำตอบ ---\n{llm_response_str}\n---")
                return None
            return json.loads(json_match.group(0))
        except json.JSONDecodeError as e:
            logging.error(f"ไม่สามารถแปลง JSON block ที่สกัดมาได้: {e}"); return None

# --- 6. Worker Function สำหรับการทำงานแบบขนาน ---
def process_case_worker(case_data: tuple) -> Dict:
    index, total, single_case_text, processor, output_dir = case_data
    
    case_number_pattern = re.compile(r'.*?\d+/\d{4}')
    case_id_pattern = re.compile(r'(\d+/\d{4})')
    
    match = case_id_pattern.search(single_case_text)
    if not match:
        return {"status": "skipped_no_id"}
        
    case_id_for_filename = match.group(1).replace('/', '-')
    case_id_for_fallback = match.group(1)
    
    output_filename = os.path.join(output_dir, f"{case_id_for_filename}.json")
    
    json_data = processor.extract_data_for_single_case(single_case_text)
    
    if json_data:
        ai_case_number = json_data.get("case_number")
        
        # ตรวจสอบความถูกต้องของ case_number ที่ AI ให้มา
        is_valid = isinstance(ai_case_number, str) and case_number_pattern.fullmatch(ai_case_number)
        
        if not is_valid:
            logging.warning(f"คดี {case_id_for_filename}: AI ให้ case_number ที่ไม่ถูกต้อง ('{ai_case_number}'). ใช้ค่า fallback '{case_id_for_fallback}' แทน")
            json_data['case_number'] = case_id_for_fallback
        
        try:
            LegalCase.model_validate(json_data)
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
            return {"status": "success", "case_number": case_id_for_filename}
        except ValidationError as e:
            logging.error(f"คดี {case_id_for_filename}: ข้อมูลไม่สมบูรณ์ (Pydantic Error) - {e}")
            return {"status": "failed", "case_number": case_id_for_filename}
    else:
        logging.error(f"คดี {case_id_for_filename}: ประมวลผลล้มเหลว ไม่ได้รับข้อมูล JSON จาก AI")
        return {"status": "failed", "case_number": case_id_for_filename}

# --- 7. ฟังก์ชันหลักในการควบคุมการทำงาน ---

def run_processing(args, output_dir):
    """ฟังก์ชันสำหรับประมวลผลไฟล์ .md และเรียก API"""
    logging.info("--- เริ่มโหมดประมวลผลไฟล์ .md ---")
    api_key = os.getenv("OPENROUTER_API_KEY")
    try:
        connector = OpenRouterConnector(api_key)
        processor = LegalCaseProcessor(connector)
    except ValueError as e: 
        logging.error(e); sys.exit(1)

    input_dir = "db_messymd"
    if not os.path.isdir(input_dir):
        logging.error(f"ไม่พบโฟลเดอร์ Input: '{input_dir}'")
        return

    existing_case_ids = set()
    if not args.force:
        logging.info(f"กำลังสแกนหาไฟล์ JSON ที่มีอยู่แล้วใน '{output_dir}'...")
        for filename in os.listdir(output_dir):
            if filename.endswith(".json"):
                existing_case_ids.add(filename.replace('.json', '').replace('-', '/'))
        logging.info(f"พบไฟล์ที่ประมวลผลแล้ว {len(existing_case_ids)} คดี จะทำการข้าม")
    else:
        logging.warning("โหมด --force: จะประมวลผลทุกคดีใหม่ทั้งหมด")

    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    all_raw_text = ""
    md_files = [f for f in os.listdir(input_dir) if f.endswith(".md")]
    if not md_files:
        logging.warning(f"ไม่พบไฟล์ .md ใน '{input_dir}'"); return

    logging.info(f"พบ {len(md_files)} ไฟล์ .md. กำลังอ่านข้อมูล...")
    for filename in sorted(md_files, key=natural_sort_key):
        try:
            with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
                all_raw_text += f.read() + "\n___________________________\n"
        except Exception as e:
            logging.error(f"เกิดข้อผิดพลาดขณะอ่าน '{filename}': {e}")
    logging.info("อ่านไฟล์ .md ทั้งหมดแล้ว")

    all_cases_in_files = all_raw_text.split("___________________________")
    case_pattern = re.compile(r'(\d+/\d{4})')
    cases_to_process = []
    
    logging.info("กำลังคัดกรองเฉพาะคดีที่ยังไม่เคยประมวลผล...")
    for case_text in all_cases_in_files:
        if not case_text.strip(): continue
        match = case_pattern.search(case_text)
        if match:
            if match.group(1) not in existing_case_ids:
                cases_to_process.append(case_text.strip())
    
    if not cases_to_process:
        logging.info("ไม่พบคดีใหม่ที่ต้องประมวลผล")
        return

    logging.info(f"พบเอกสารคดีใหม่ {len(cases_to_process)} คดี. เริ่มการประมวลผลแบบขนาน...")
    tasks = [(i + 1, len(cases_to_process), text, processor, output_dir) for i, text in enumerate(cases_to_process)]
    counts = {"success": 0, "failed": 0, "skipped_no_id": 0}
    max_workers = min(16, (os.cpu_count() or 1) + 4)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(process_case_worker, task): task for task in tasks}
        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="กำลังประมวลผลคดีใหม่"):
            try:
                result = future.result()
                if result and result.get("status") in counts:
                    counts[result["status"]] += 1
            except Exception as exc:
                logging.error(f'คดีที่ {future_to_task[future][0]} เกิดข้อผิดพลาดรุนแรง: {exc}')
                counts["failed"] += 1
    
    logging.info("--- สรุปผลการประมวลผลรอบนี้ ---")
    logging.info(f"สำเร็จ (สร้างไฟล์ใหม่): {counts['success']} คดี")
    logging.info(f"ล้มเหลว: {counts['failed']} คดี")
    logging.info(f"ข้าม (หาเลขคดีไม่เจอ): {counts['skipped_no_id']} คดี")

def run_verification_and_repair(output_dir):
    """ฟังก์ชันสำหรับตรวจสอบและซ่อมแซม case_number ในไฟล์ JSON ทั้งหมด"""
    logging.info("--- เริ่มโหมดตรวจสอบและซ่อมแซมไฟล์ JSON ---")
    
    files_to_check = [f for f in os.listdir(output_dir) if f.endswith(".json") and f != "_ALL_CASES_COMBINED.json"]
    if not files_to_check:
        logging.warning(f"ไม่พบไฟล์ JSON ที่จะตรวจสอบใน '{output_dir}'")
        return

    logging.info(f"พบ {len(files_to_check)} ไฟล์ JSON ที่จะทำการตรวจสอบ...")
    
    validation_pattern = re.compile(r'.*?\d+/\d{4}')
    repaired_count = 0
    valid_count = 0
    
    for filename in tqdm(files_to_check, desc="กำลังตรวจสอบไฟล์ JSON"):
        filepath = os.path.join(output_dir, filename)
        try:
            with open(filepath, 'r+', encoding='utf-8') as f:
                data = json.load(f)
                current_case_number = data.get("case_number", "")

                if isinstance(current_case_number, str) and validation_pattern.fullmatch(current_case_number):
                    valid_count += 1
                    continue
                
                new_case_number = filename.replace('.json', '').replace('-', '/')
                data['case_number'] = new_case_number
                
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=4)
                f.truncate()
                repaired_count += 1

        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"ไม่สามารถตรวจสอบหรือแก้ไขไฟล์ '{filename}': {e}")
    
    logging.info("--- สรุปผลการตรวจสอบ ---")
    logging.info(f"ไฟล์ที่ถูกต้องอยู่แล้ว: {valid_count} ไฟล์")
    logging.info(f"ไฟล์ที่ถูกซ่อมแซม case_number: {repaired_count} ไฟล์")

def update_combined_file(output_dir):
    """ฟังก์ชันสำหรับสร้างไฟล์ _ALL_CASES_COMBINED.json ขึ้นมาใหม่เสมอ"""
    logging.info("กำลังอัปเดตไฟล์สรุปรวม _ALL_CASES_COMBINED.json...")
    all_successful_cases = []
    combined_filename_leaf = "_ALL_CASES_COMBINED.json"
    
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s.replace(".json", ""))]

    file_list = [f for f in os.listdir(output_dir) if f.endswith(".json") and f != combined_filename_leaf]

    for filename in sorted(file_list, key=natural_sort_key):
        try:
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                if os.path.getsize(filepath) > 0:
                    all_successful_cases.append(json.load(f))
                else:
                    logging.warning(f"ข้ามไฟล์สรุป '{filename}' เนื่องจากเป็นไฟล์ว่าง")
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"ไม่สามารถอ่าน '{filename}' เพื่อนำมารวมได้: {e}")

    if all_successful_cases:
        combined_filepath = os.path.join(output_dir, combined_filename_leaf)
        with open(combined_filepath, "w", encoding="utf-8") as f:
            json.dump(all_successful_cases, f, ensure_ascii=False, indent=4)
        logging.info(f"อัปเดตไฟล์สรุปรวมสำเร็จ มีทั้งหมด {len(all_successful_cases)} คดี")

def main():
    parser = argparse.ArgumentParser(description="ประมวลผลและตรวจสอบไฟล์กฎหมาย")
    parser.add_argument('--force', action='store_true', help='บังคับให้ประมวลผลทุกคดีใหม่ทั้งหมด')
    parser.add_argument('--verify-only', action='store_true', help='ทำงานในโหมดตรวจสอบและซ่อมแซมไฟล์ JSON ที่มีอยู่เท่านั้น')
    args = parser.parse_args()

    start_time = time.time()
    load_dotenv()
    output_dir = "processed_cases"
    os.makedirs(output_dir, exist_ok=True)

    if args.verify_only:
        run_verification_and_repair(output_dir)
    else:
        run_processing(args, output_dir)
        run_verification_and_repair(output_dir) # ตรวจสอบซ้ำอีกครั้งหลังประมวลผลเสร็จ
    
    update_combined_file(output_dir)

    end_time = time.time()
    logging.info(f"🎉 กระบวนการทั้งหมดเสร็จสมบูรณ์! ใช้เวลา: {end_time - start_time:.2f} วินาที")

if __name__ == "__main__":
    main()