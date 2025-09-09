import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from starlette.concurrency import run_in_threadpool
import logging
import uvicorn

# Proje dosyalarını import et
from run_pipeline import get_pipeline
from intent_parser import parse_intent
from sozluk_duzeltici import sozlukDuzelt2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sesli Asistan API Servisi")

# Modeli CPU için en verimli ayarlarla yapılandır
MODEL_CONFIG = {
    "modelSize": "medium",
    "device": "cpu",
    "computeType": "int8"
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

@app.websocket("/ws")
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

@app.post("/audio") # <-- YOL GÜNCELLENDİ
async def process_audio_file(file: UploadFile = File(...)):
    logger.info(f"Nihai ses dosyası /audio endpoint'ine geldi: {file.filename}")
    contents = await file.read()
    result = pipeline.process_audio_from_file_upload(contents, file.filename)
    logger.info(f"Nihai Sonuç: {result}")
    return result

if __name__ == "__main__":
    # Yeni IP ve Port ayarları
    host_ip = "10.20.30.10"
    port_num = 8080

    print(f"Sesli Asistan API servisi başlatılıyor... Model: {MODEL_CONFIG['modelSize']}, Cihaz: {MODEL_CONFIG['device']}, Hesaplama: {MODEL_CONFIG['computeType']}")
    # Bilgilendirme mesajı yeni adreslerle güncellendi
    print(f"API endpointleri: ws://{host_ip}:{port_num}/ws ve http://{host_ip}:{port_num}/audio")
    
    # Uvicorn yeni host ve port ile çalıştırılıyor
    uvicorn.run("server:app", host=host_ip, port=port_num, reload=True)