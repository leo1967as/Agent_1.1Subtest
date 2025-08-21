# chroma_manager.py
import os
import pickle
import logging
import chromadb
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, ValidationError
import torch
# --- 1. ตั้งค่าระบบบันทึก Log และ Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# --- Configuration ---
DB_PATH = "legal_vector_db"
COLLECTION_NAME = "legal_cases_bge_m3"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDINGS_PICKLE_PATH = "embeddings_bge_m3.pkl"

# --- 2. Pydantic Models สำหรับตรวจสอบข้อมูล Input ---
# (ไม่จำเป็นต้องใช้โดยตรง แต่มีไว้เพื่อความชัดเจนของโครงสร้างข้อมูล)
class EmbeddingData(BaseModel):
    case_numbers: List[str]
    texts: List[str]
    embeddings: np.ndarray

    class Config:
        arbitrary_types_allowed = True

# --- 3. คลาสจัดการ ChromaDB (หัวใจของสคริปต์) ---
class ChromaDBManager:
    """
    คลาสที่จัดการทุกอย่างเกี่ยวกับ ChromaDB Vector Store อย่างสมบูรณ์
    """
    def __init__(self, db_path: str, collection_name: str):
        """
        Constructor: ทำการเชื่อมต่อหรือสร้างฐานข้อมูลและ Collection
        """
        try:
            # Line-by-Line: สร้าง Client แบบ Persistent ซึ่งจะบันทึกข้อมูลลงในโฟลเดอร์ที่ระบุ
            self.client = chromadb.PersistentClient(path=db_path)
            
            # Line-by-Line: เรียกใช้หรือสร้าง Collection ขึ้นมาใหม่
            # metadata={"hnsw:space": "cosine"} คือการบอกให้ใช้ Cosine Similarity ในการคำนวณความคล้ายคลึง ซึ่งเหมาะกับ bge-m3
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"} 
            )
            logging.info(f"เชื่อมต่อกับ ChromaDB ที่ '{db_path}' และ Collection '{collection_name}' สำเร็จ")
        except Exception as e:
            logging.error(f"ไม่สามารถเริ่มต้น ChromaDB ได้: {e}")
            raise

    def get_collection_status(self) -> Dict:
        """
        ตรวจสอบสถานะปัจจุบันของ Collection (เช่น จำนวนรายการ)
        """
        try:
            count = self.collection.count()
            return {"item_count": count}
        except Exception as e:
            logging.error(f"ไม่สามารถดึงสถานะของ Collection ได้: {e}")
            return {"item_count": -1}

    def is_id_exist(self, doc_id: str) -> bool:
        """
        ตรวจสอบว่ามีเอกสารที่มี ID ที่ระบุอยู่ใน Collection แล้วหรือไม่
        """
        try:
            result = self.collection.get(ids=[doc_id])
            return len(result['ids']) > 0
        except Exception as e:
            logging.error(f"เกิดข้อผิดพลาดขณะตรวจสอบ ID '{doc_id}': {e}")
            return False # สมมติว่าไม่มีอยู่เพื่อความปลอดภัย

    def add_data(self, embeddings: np.ndarray, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """
        เพิ่มข้อมูลใหม่เข้าไปใน Collection พร้อมจัดการข้อมูลซ้ำซ้อนและ Batching
        """
        if not all([isinstance(embeddings, np.ndarray), isinstance(documents, list), isinstance(metadatas, list), isinstance(ids, list)]):
            logging.error("ประเภทข้อมูล Input ไม่ถูกต้อง")
            return

        if not (len(embeddings) == len(documents) == len(metadatas) == len(ids)):
            logging.error("จำนวนรายการของข้อมูลแต่ละส่วนไม่เท่ากัน")
            return

        # --- ส่วนจัดการข้อมูลซ้ำซ้อน ---
        new_embeddings, new_documents, new_metadatas, new_ids = [], [], [], []
        for emb, doc, meta, doc_id in zip(embeddings, documents, metadatas, ids):
            if not self.is_id_exist(doc_id):
                new_embeddings.append(emb.tolist()) # ChromaDB ต้องการ List of floats
                new_documents.append(doc)
                new_metadatas.append(meta)
                new_ids.append(doc_id)
            else:
                logging.warning(f"ข้ามการเพิ่มข้อมูล ID '{doc_id}' เนื่องจากมีอยู่แล้ว")
        
        if not new_ids:
            logging.info("ไม่มีข้อมูลใหม่ที่จะเพิ่ม")
            return

        logging.info(f"กำลังจะเพิ่มข้อมูลใหม่จำนวน {len(new_ids)} รายการ...")
        
        try:
            # Line-by-Line: ChromaDB จัดการ Batching ให้อัตโนมัติเพื่อประสิทธิภาพสูงสุด
            self.collection.add(
                embeddings=new_embeddings,
                documents=new_documents,
                metadatas=new_metadatas,
                ids=new_ids
            )
            logging.info(f"เพิ่มข้อมูลใหม่ {len(new_ids)} รายการสำเร็จ")
        except Exception as e:
            logging.error(f"เกิดข้อผิดพลาดร้ายแรงขณะเพิ่มข้อมูลลงใน ChromaDB: {e}")

    def query(self, query_embedding: np.ndarray, n_results: int = 5) -> Optional[Dict]:
        """
        ค้นหาข้อมูลจาก Vector Embedding ที่ให้มา
        """
        try:
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results
            )
            return results
        except Exception as e:
            logging.error(f"เกิดข้อผิดพลาดขณะทำการค้นหา: {e}")
            return None

    def clear_collection(self):
        """
        ลบข้อมูลทั้งหมดใน Collection (ใช้ด้วยความระมัดระวัง!)
        """
        logging.warning(f"!!! กำลังจะลบข้อมูลทั้งหมดใน Collection '{self.collection.name}' !!!")
        try:
            count = self.collection.count()
            if count > 0:
                all_ids = self.collection.get(limit=count)['ids']
                self.collection.delete(ids=all_ids)
                logging.info(f"ลบข้อมูลทั้งหมด {count} รายการออกจาก Collection '{self.collection.name}' สำเร็จ")
            else:
                logging.info("Collection ว่างอยู่แล้ว ไม่มีการดำเนินการใดๆ")
        except Exception as e:
            logging.error(f"ไม่สามารถล้างข้อมูลใน Collection ได้: {e}")

# --- 4. ส่วนของการรันโปรแกรมหลัก ---
def main():
    # --- โหลดข้อมูล Embeddings ---
    try:
        with open(EMBEDDINGS_PICKLE_PATH, "rb") as f_in:
            data = pickle.load(f_in)
        
        # ตรวจสอบข้อมูลด้วย Pydantic (ทางเลือก แต่แนะนำ)
        embedding_data = EmbeddingData.model_validate(data)
        logging.info(f"โหลดไฟล์ '{EMBEDDINGS_PICKLE_PATH}' สำเร็จ")
    except (FileNotFoundError, ValidationError, pickle.UnpicklingError) as e:
        logging.error(f"ไม่สามารถโหลดหรือตรวจสอบไฟล์ '{EMBEDDINGS_PICKLE_PATH}' ได้: {e}")
        return

    # --- เริ่มต้นการใช้งาน ChromaDBManager ---
    try:
        db_manager = ChromaDBManager(db_path=DB_PATH, collection_name=COLLECTION_NAME)
    except Exception:
        logging.error("ไม่สามารถเริ่มต้น ChromaDBManager ได้ สิ้นสุดการทำงาน")
        return

    # --- แสดงสถานะก่อนเพิ่มข้อมูล ---
    status_before = db_manager.get_collection_status()
    logging.info(f"สถานะปัจจุบัน: มี {status_before.get('item_count', 'N/A')} รายการใน Collection")

    # --- เตรียมข้อมูลและเพิ่มลงใน DB ---
    embeddings = embedding_data.embeddings
    documents = embedding_data.texts
    case_numbers = embedding_data.case_numbers
    
    metadatas = [{"case_number": cn} for cn in case_numbers]
    ids = [f"case_{cn.replace('/', '-')}" for cn in case_numbers]

    db_manager.add_data(embeddings, documents, metadatas, ids)

    # --- แสดงสถานะหลังเพิ่มข้อมูล ---
    status_after = db_manager.get_collection_status()
    logging.info(f"สถานะหลังเพิ่มข้อมูล: มี {status_after.get('item_count', 'N/A')} รายการใน Collection")

    # --- ทดสอบการค้นหา ---
    if status_after.get('item_count', 0) > 0:
        logging.info("\n--- เริ่มทดสอบการค้นหา ---")
        
        # โหลดโมเดลสำหรับสร้าง Embedding ของคำค้นหา
        logging.info(f"กำลังโหลดโมเดล '{EMBEDDING_MODEL_NAME}' สำหรับการค้นหา...")
        query_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        query_text = "คดีพิพาทเรื่องการรุกล้ำที่ดินสาธารณะโดยหน่วยงานราชการ"
        logging.info(f"คำค้นหา: '{query_text}'")
        
        query_embedding = query_model.encode([query_text], normalize_embeddings=True)
        
        search_results = db_manager.query(query_embedding, n_results=3)
        
        if search_results and search_results['documents']:
            logging.info("ผลการค้นหา 3 อันดับแรก:")
            for i, (doc, dist) in enumerate(zip(search_results['documents'][0], search_results['distances'][0])):
                case_num = search_results['metadatas'][0][i]['case_number']
                logging.info(f"  อันดับ {i+1}: เลขคดี {case_num} (ความคล้ายคลึง: {1-dist:.4f})")
                logging.info(f"    เนื้อหา: {doc[:200]}...") # แสดง 200 ตัวอักษรแรก
        else:
            logging.warning("ไม่พบผลลัพธ์จากการค้นหา")

    # --- ตัวอย่างการล้างข้อมูล (ปิดไว้เป็นค่าเริ่มต้น) ---
    # Uncomment บรรทัดด้านล่างถ้าต้องการทดสอบการล้างข้อมูลทั้งหมด
    # logging.info("\n--- เริ่มทดสอบการล้างข้อมูล ---")
    # db_manager.clear_collection()
    # status_cleared = db_manager.get_collection_status()
    # logging.info(f"สถานะหลังล้างข้อมูล: มี {status_cleared.get('item_count', 'N/A')} รายการใน Collection")

if __name__ == "__main__":
    main()