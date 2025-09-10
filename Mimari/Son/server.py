# server.py

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
# YENİ: CORSMiddleware'i import edin
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
import logging
import uvicorn

# Proje dosyalarını import et
from run_pipeline import get_pipeline, process_audio_from_file_upload # process_audio_from_file_upload'ı run_pipeline'dan alıyoruz
from intent_parser import parse_intent
from sozluk_duzeltici import sozlukDuzelt2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sesli Asistan API Servisi")

# ================= CORS AYARLARI =================
# Geliştirme ortamı için localhost olabilir.
# Production için "http://uygulamanizin-adresi.com" gibi olmalı.
origins = [
    "http://localhost",
    "http://localhost:3000", 
    "http://localhost:8080", 
    "http://10.20.30.20:8080", 
    
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Belirtilen kaynaklardan gelen isteklere izin ver
    allow_credentials=True, # Çerezlere izin ver (gerekirse)
    allow_methods=["*"],    # Tüm metotlara (GET, POST, vb.) izin ver
    allow_headers=["*"],    # Tüm başlıklara (header) izin ver
)
# ==================================================

# Modeli CPU için en verimli ayarlarla yapılandır
MODEL_CONFIG = {
    "modelSize": "medium",
    "device": "cpu",
    "computeType": "int8",
    "memoryWarningGB": 3.0,
    "memoryRestartGB": 8.0
}
# Pipeline'ı ve modeli sunucu başlarken sadece bir kere yükle
pipeline = get_pipeline(model_config=MODEL_CONFIG)
model = pipeline.transcription_engine.model

def process_chunk_sync(audio_chunk_np: np.ndarray):
    segments, info = model.transcribe(audio_chunk_np, language="tr", task="transcribe")
    text = "".join(seg.text for seg in segments).strip()
    if not text:
        return None
    corrected_text = sozlukDuzelt2(text)
    intent = parse_intent(corrected_text)
    return {
        "type": "interim_result",
        "transcript": corrected_text,
        "intent": intent
    }

@app.websocket("/ws")   # BURADAKİ ADRESİ ANLAŞILAN KISMA GÖRE DEĞİŞTİR!
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket bağlantısı kabul edildi.")
    audio_buffer = bytearray()
    TRANSCRIPTION_THRESHOLD_BYTES = 64000
    try:
        while True:
            data = await websocket.receive_bytes()
            audio_buffer.extend(data)
            if len(audio_buffer) >= TRANSCRIPTION_THRESHOLD_BYTES:
                processing_buffer = bytes(audio_buffer)
                audio_buffer.clear()
                audio_np = np.frombuffer(processing_buffer, dtype=np.float32)
                result = await run_in_threadpool(process_chunk_sync, audio_np)
                if result:
                    logger.info(f"Anlık Sonuç: {result['transcript']}")
                    await websocket.send_json(result)
    except WebSocketDisconnect:
        logger.info("WebSocket bağlantısı kapandı.")
    except Exception as e:
        logger.error(f"WebSocket hatası: {e}", exc_info=True)

@app.post("/audio")    # http://192.168.1.35:8080/audio => IP adresi değişecek!
async def process_audio_file(file: UploadFile = File(...)):
    logger.info(f"Nihai ses dosyası /audio endpoint'ine geldi: {file.filename}")
    try:
        contents = await file.read()
        # Bu fonksiyonun `run_pipeline.py` içinde tanımlı olduğunu varsayıyorum
        result = await run_in_threadpool(process_audio_from_file_upload, contents, file.filename)
        logger.info(f"Nihai Sonuç: {result}")
        return result
    except Exception as e:
        logger.error(f"/audio endpoint'inde hata: {e}", exc_info=True)
        # Hata durumunda istemciye anlamlı bir cevap dön
        return {"error": "Ses dosyası işlenirken bir hata oluştu.", "details": str(e)}

if __name__ == "__main__":
    host_ip = "192.168.1.35"
    port_num = 8080
    print(f"Sesli Asistan API servisi başlatılıyor... Model: {MODEL_CONFIG['modelSize']}, Cihaz: {MODEL_CONFIG['device']}, Hesaplama: {MODEL_CONFIG['computeType']}")
    print(f"API endpointleri: ws://{host_ip}:{port_num}/ws ve http://{host_ip}:{port_num}/audio")
    uvicorn.run("server:app", host=host_ip, port=port_num, reload=True)