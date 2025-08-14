import pyaudio
import wave
import threading
import queue
import whisper
import numpy as np
import time
from collections import deque
import os

class RealtimeWhisperSTT:
    def __init__(self, model_size="base", language="tr", device="cpu"):
        """
        Real-time Speech-to-Text sınıfı
        
        Args:
            model_size (str): Whisper model boyutu ("tiny", "base", "small", "medium", "large", "large-v2", "large-v3")
            language (str): Dil kodu (Türkçe için "tr")
            device (str): Cihaz ("cpu" veya "cuda")
        """
        print(f"Whisper modeli yükleniyor: {model_size}")
        self.model = whisper.load_model(model_size, device=device)
        self.language = language
        
        # Ses ayarları
        self.CHUNK = 1024  # Ses buffer boyutu
        self.FORMAT = pyaudio.paInt16  # 16-bit ses formatı
        self.CHANNELS = 1  # Mono ses
        self.RATE = 16000  # Örnekleme hızı (Whisper için optimize)
        
        # Buffer ayarları
        self.audio_buffer = deque(maxlen=int(self.RATE * 30))  # 30 saniye buffer
        self.transcription_queue = queue.Queue()
        
        # Threading kontrolü
        self.is_recording = False
        self.is_transcribing = False
        
        # PyAudio başlatma
        self.audio = pyaudio.PyAudio()
        
        print("Real-time STT sistemi hazır!")
    
    def start_recording(self):
        """Ses kaydını başlatır"""
        try:
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                stream_callback=self._audio_callback
            )
            
            self.is_recording = True
            self.stream.start_stream()
            print("🎤 Kayıt başladı. Konuşmaya başlayabilirsiniz...")
            
        except Exception as e:
            print(f"❌ Ses kaydı başlatılamadı: {e}")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Ses callback fonksiyonu - ses verilerini buffer'a ekler"""
        if self.is_recording:
            # Ses verisini numpy array'e çevir
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            # Buffer'a ekle
            self.audio_buffer.extend(audio_data)
        
        return (in_data, pyaudio.paContinue)
    
    def start_transcription(self, segment_duration=5):
        """
        Transkripsiyonu başlatır
        
        Args:
            segment_duration (int): Her segment için saniye cinsinden süre
        """
        self.segment_duration = segment_duration
        self.segment_samples = int(self.RATE * segment_duration)
        self.is_transcribing = True
        
        # Transkripsiyon thread'ini başlat
        transcription_thread = threading.Thread(target=self._transcription_worker)
        transcription_thread.daemon = True
        transcription_thread.start()
        
        print(f"📝 Transkripsiyon başladı ({segment_duration} saniye segmentler)")
    
    def _transcription_worker(self):
        """Transkripsiyon işçi thread'i"""
        last_transcription_time = time.time()
        
        while self.is_transcribing:
            current_time = time.time()
            
            # Segment süresi geçtiyse ve yeterli ses verisi varsa
            if (current_time - last_transcription_time >= self.segment_duration and 
                len(self.audio_buffer) >= self.segment_samples):
                
                # Son segment'i al
                audio_segment = np.array(list(self.audio_buffer)[-self.segment_samples:])
                
                # Ses seviyesi kontrol et (sessizlik kontrolü)
                if self._is_speech(audio_segment):
                    try:
                        # Whisper için ses verisini normalize et
                        audio_float = audio_segment.astype(np.float32) / 32768.0
                        
                        # Whisper ile transkripsiyon yap
                        result = self.model.transcribe(
                            audio_float, 
                            language=self.language,
                            task="transcribe",
                            verbose=False
                        )
                        
                        transcription = result["text"].strip()
                        
                        if transcription:
                            # Zaman damgası ile birlikte queue'ya ekle
                            timestamp = time.strftime("%H:%M:%S")
                            self.transcription_queue.put((timestamp, transcription))
                    
                    except Exception as e:
                        print(f"❌ Transkripsiyon hatası: {e}")
                
                last_transcription_time = current_time
            
            time.sleep(0.1)  # CPU kullanımını azalt
    
    def _is_speech(self, audio_data, threshold=50):
        """
        Ses verisinin konuşma içerip içermediğini kontrol eder
        
        Args:
            audio_data (np.array): Ses verisi
            threshold (int): Ses seviyesi eşiği
        
        Returns:
            bool: Konuşma varsa True
        """
        # RMS (Root Mean Square) hesapla
        rms = np.sqrt(np.mean(audio_data**2))
        return rms > threshold
    
    def get_transcription(self):
        """Bekleyen transkripsiyonları döndürür"""
        transcriptions = []
        while not self.transcription_queue.empty():
            transcriptions.append(self.transcription_queue.get())
        return transcriptions
    
    def stop_recording(self):
        """Ses kaydını durdurur"""
        if hasattr(self, 'stream') and self.stream.is_active():
            self.is_recording = False
            self.stream.stop_stream()
            self.stream.close()
            print("🛑 Kayıt durduruldu")
    
    def stop_transcription(self):
        """Transkripsiyonu durdurur"""
        self.is_transcribing = False
        print("📝 Transkripsiyon durduruldu")
    
    def cleanup(self):
        """Kaynakları temizler"""
        self.stop_recording()
        self.stop_transcription()
        self.audio.terminate()
        print("✅ Sistem temizlendi")

def main():
    """Ana fonksiyon - STT sistemini çalıştırır"""
    print("=== Real-time Whisper Speech-to-Text ===\n")
    
    # STT sistemini başlat
    # Model seçenekleri: "tiny", "base", "small", "medium", "large", "large-v3"
    # GPU varsa device="cuda" kullanın
    stt = RealtimeWhisperSTT(
        model_size="base",  # Hız için "base", kalite için "large" kullanın
        language="tr",      # Türkçe
        device="cpu"        # GPU varsa "cuda" yapın
    )
    
    try:
        # Kayıt ve transkripsiyonu başlat
        stt.start_recording()
        stt.start_transcription(segment_duration=3)  # 3 saniye segmentler
        
        print("\n📢 Komutlar:")
        print("- Konuşun, metinler otomatik görünecek")
        print("- 'quit' yazıp Enter'a basarak çıkın\n")
        
        # Ana döngü
        while True:
            # Yeni transkripsiyonları al ve göster
            transcriptions = stt.get_transcription()
            for timestamp, text in transcriptions:
                print(f"[{timestamp}] {text}")
            
            # Kullanıcı komutlarını kontrol et
            try:
                # Non-blocking input check (Windows için alternatif gerekebilir)
                import select
                import sys
                
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    user_input = input().strip().lower()
                    if user_input == 'quit':
                        break
            except:
                # Windows'ta select çalışmazsa basit input kullan
                pass
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\n🛑 Kullanıcı tarafından durduruldu")
    
    finally:
        # Temizlik
        stt.cleanup()
        print("👋 Program sonlandırıldı")

if __name__ == "__main__":
    main()