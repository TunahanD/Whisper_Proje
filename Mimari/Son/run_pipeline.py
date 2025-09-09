# =============================
# FILE: run_pipeline.py - REFACTORED
# =============================
"""
Etkileşimli Transkripsiyon ve Niyet Analizi Pipeline'ı
- Model bir kez yüklenir, memory'de kalır
- Temiz API ile modüler yapı
- CLI ve programmatic interface ayrımı
"""

import os
import sys
from main import TranscriptionEngine
from intent_parser import parse_intent, prepare_output_folder, save_to_file

class AudioProcessingPipeline:
    def __init__(self, model_config=None):
        """
        Pipeline'ı başlat ve modeli yükle
        """
        print("="*50)
        print("Etkileşimli Transkripsiyon ve Niyet Analizi Sistemi")
        print("="*50)
        
        # Transkripsiyon engine'ini başlat
        print("\n[Sistem] Whisper modeli belleğe yükleniyor...")
        self.transcription_engine = TranscriptionEngine(model_config)
        self.transcription_engine.load_model()  # Model'i şimdi yükle
        print("[Sistem] ✅ Model başarıyla yüklendi ve kullanıma hazır.")
        
        # Çıktı klasörünü hazırla
        prepare_output_folder()
        self.file_counter = 1

    def process_audio_file(self, audio_path: str, save_to_json: bool = True) -> dict:
        """
        Tek bir ses dosyasını işler
        Returns: intent_result (sadece niyet analizi sonucu)
        """
        print(f"\n[Pipeline] Ses dosyası işleniyor: {audio_path}")
        
        # Dosya varlığını kontrol et
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Dosya bulunamadı: {audio_path}")
        
        # 1. Transkripsiyon
        print("[Pipeline] Transkripsiyon başlıyor...")
        transcription_result = self.transcription_engine.transcribe_file(audio_path)
        
        if not transcription_result["corrected_text"].strip():
            raise ValueError("Transkripsiyon başarısız - ses dosyasında konuşma tespit edilemedi")
        
        print(f"[Pipeline] Transkript: {transcription_result['corrected_text']}")
        
        # 2. Niyet Analizi
        print("[Pipeline] Niyet analizi yapılıyor...")
        intent_result = parse_intent(transcription_result["corrected_text"])
        
        # 3. JSON olarak kaydet (sadece intent analysis)
        if save_to_json:
            output_filename = f"output/intent_{self.file_counter}.json"
            save_to_file(intent_result, filename=output_filename)
            print(f"[Pipeline] Sonuç kaydedildi: {output_filename}")
            self.file_counter += 1
        
        return intent_result

    def interactive_mode(self):
        """
        Kullanıcıdan sürekli dosya yolu alarak işler
        """
        while True:
            print("\n" + "-"*50)
            audio_path = input("WAV dosya yolu girin (çıkmak için 'exit'): ").strip()
            
            if audio_path.lower() == 'exit':
                print("\n[Sistem] Programdan çıkılıyor. Hoşça kalın!")
                break
            
            try:
                result = self.process_audio_file(audio_path)
                print("\n✅ İşlem başarıyla tamamlandı!")
                
            except FileNotFoundError as e:
                print(f"[HATA] {e}")
            except ValueError as e:
                print(f"[HATA] {e}")
            except Exception as e:
                print(f"[HATA] Beklenmeyen hata: {e}")

    def batch_process(self, file_list: list[str]) -> list[dict]:
        """
        Birden fazla dosyayı toplu işler
        """
        results = []
        for audio_path in file_list:
            try:
                result = self.process_audio_file(audio_path)
                results.append(result)
                print(f"✅ {audio_path} işlendi")
            except Exception as e:
                print(f"❌ {audio_path} işlenemedi: {e}")
                results.append({"error": str(e), "file": audio_path})
        
        return results

# =============================
# Flask/Web API için fonksiyon
# =============================
def create_pipeline_instance(model_config=None):
    """
    Flask uygulaması için pipeline instance'ı oluştur
    """
    return AudioProcessingPipeline(model_config)

# Global pipeline instance (lazy initialization)
_pipeline = None

def get_pipeline():
    """Singleton pattern"""
    global _pipeline
    if _pipeline is None:
        _pipeline = AudioProcessingPipeline()
    return _pipeline

def process_audio_from_memory(audio_bytes_io, original_filename: str = "audio") -> dict:
    """
    Flask için BytesIO objesi alan versiyon
    audio_bytes_io: io.BytesIO objesi
    original_filename: Orijinal dosya adı (format tespiti için)
    """
    pipeline = get_pipeline()
    
    # BytesIO'dan byte array'i al
    audio_bytes = audio_bytes_io.getvalue()
    
    # TranscriptionEngine'in transcribe_from_bytes metodunu kullan
    transcription_result = pipeline.transcription_engine.transcribe_from_bytes(
        audio_bytes, 
        original_filename
    )
    
    print(f"[Pipeline] Transkript: {transcription_result['corrected_text']}")
    
    # Niyet analizi
    print("[Pipeline] Niyet analizi yapılıyor...")
    intent_result = parse_intent(transcription_result["corrected_text"])
    
    return intent_result

def process_audio_from_file_upload(file_content: bytes, filename: str) -> dict:
    """
    Web upload için direkt byte content alan versiyon
    file_content: Dosyanın byte içeriği
    filename: Upload edilen dosyanın orijinal adı
    """
    pipeline = get_pipeline()
    
    # TranscriptionEngine'in transcribe_from_bytes metodunu kullan
    transcription_result = pipeline.transcription_engine.transcribe_from_bytes(
        file_content, 
        filename
    )
    
    print(f"[Pipeline] Transkript: {transcription_result['corrected_text']}")
    
    # Niyet analizi
    print("[Pipeline] Niyet analizi yapılıyor...")
    intent_result = parse_intent(transcription_result["corrected_text"])
    
    return intent_result

# =============================
# CLI Interface
# =============================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Audio Processing Pipeline")
    parser.add_argument('--model-size', type=str, default="medium")
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--batch', type=str, nargs='+', help='Toplu işleme için dosya listesi')
    
    args = parser.parse_args()
    
    # Model konfigürasyonu
    model_config = {
        "modelSize": args.model_size,
        "device": args.device
    }
    
    # Pipeline'ı başlat
    pipeline = AudioProcessingPipeline(model_config)
    
    if args.batch:
        # Toplu işleme modu
        print(f"\n[Batch] {len(args.batch)} dosya işlenecek...")
        results = pipeline.batch_process(args.batch)
        print(f"\n[Batch] İşlem tamamlandı. {len(results)} sonuç.")
    else:
        # Etkileşimli mod
        pipeline.interactive_mode()

if __name__ == "__main__":
    main()