import pyaudio
import wave
import threading
import queue
import whisper
import numpy as np
import time
from collections import deque
import os

class VADWhisperSTT:
    def __init__(self, model_size="base", language="tr", device="cpu"):
        """
        Voice Activity Detection + Whisper STT sınıfı
        
        Args:
            model_size (str): Whisper model boyutu
            language (str): Dil kodu (Türkçe için "tr")
            device (str): Cihaz ("cpu" veya "cuda")
        """
        print(f"Whisper modeli yükleniyor: {model_size}")
        self.model = whisper.load_model(model_size, device=device)
        self.language = language
        
        # Ses ayarları
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        
        # VAD ayarları
        self.speech_threshold = 200    # Konuşma algılama eşiği
        self.silence_threshold = 100   # Sessizlik eşiği
        self.min_speech_duration = 0.5  # En az konuşma süresi (saniye)
        self.silence_timeout = 2.0     # Sessizlik timeout süresi (saniye)
        
        # Durum takibi
        self.is_speaking = False       # Şu an konuşuyor mu?
        self.speech_start_time = 0     # Konuşma başlama zamanı
        self.last_speech_time = 0      # Son ses algılama zamanı
        self.current_speech_buffer = []  # Mevcut konuşma buffer'ı
        
        # Threading
        self.is_recording = False
        self.transcription_queue = queue.Queue()
        
        # PyAudio
        self.audio = pyaudio.PyAudio()
        
        print("🎤 VAD + Whisper STT sistemi hazır!")
    
    def _calculate_rms(self, audio_data):
        """Ses verisinin RMS (Root Mean Square) değerini hesaplar"""
        return np.sqrt(np.mean(audio_data.astype(np.float64)**2))
    
    def _detect_speech_activity(self, audio_chunk):
        """
        Voice Activity Detection - ses aktivitesini algılar
        
        Args:
            audio_chunk (np.array): Ses verisi
            
        Returns:
            bool: Konuşma algılandıysa True
        """
        rms = self._calculate_rms(audio_chunk)
        current_time = time.time()
        
        # Debug: Ses seviyesini göster
        status = "🔊" if rms > self.speech_threshold else "🔇"
        print(f"\r{status} Ses: {rms:4.0f} | Durum: {'KONUŞUYOR' if self.is_speaking else 'BEKLİYOR'}", end="", flush=True)
        
        # Konuşma algılama mantığı
        if rms > self.speech_threshold:
            if not self.is_speaking:
                # Yeni konuşma başlıyor
                self.is_speaking = True
                self.speech_start_time = current_time
                self.current_speech_buffer = []
                print(f"\n🎤 Konuşma başladı! (RMS: {rms:.0f})")
            
            self.last_speech_time = current_time
            return True
            
        elif self.is_speaking and rms < self.silence_threshold:
            # Sessizlik kontrolü
            silence_duration = current_time - self.last_speech_time
            
            if silence_duration > self.silence_timeout:
                # Konuşma bitti
                speech_duration = current_time - self.speech_start_time
                
                if speech_duration > self.min_speech_duration:
                    print(f"\n🛑 Konuşma bitti! (Süre: {speech_duration:.1f}s)")
                    self._process_speech()
                else:
                    print(f"\n⚠️ Çok kısa konuşma, atlandı (Süre: {speech_duration:.1f}s)")
                
                self.is_speaking = False
                self.current_speech_buffer = []
        
        return self.is_speaking
    
    def _process_speech(self):
        """Kaydedilen konuşmayı Whisper ile işler"""
        if not self.current_speech_buffer:
            print("⚠️ İşlenecek ses verisi yok")
            return
        
        print("🤖 Whisper ile transkripsiyon yapılıyor...")
        
        try:
            # Buffer'ı birleştir ve normalize et
            audio_data = np.concatenate(self.current_speech_buffer)
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Whisper transkripsiyon
            result = self.model.transcribe(
                audio_float,
                language=self.language,
                task="transcribe",
                verbose=False
            )
            
            transcription = result["text"].strip()
            
            if transcription:
                timestamp = time.strftime("%H:%M:%S")
                self.transcription_queue.put((timestamp, transcription))
                print(f"✅ Transkripsiyon: {transcription}")
            else:
                print("⚠️ Boş transkripsiyon sonucu")
                
        except Exception as e:
            print(f"❌ Transkripsiyon hatası: {e}")
    
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
            print("🎤 VAD kayıt sistemi başladı!")
            print(f"📊 Ayarlar: Konuşma>{self.speech_threshold}, Sessizlik<{self.silence_threshold}, Timeout>{self.silence_timeout}s")
            
        except Exception as e:
            print(f"❌ Ses kaydı başlatılamadı: {e}")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Ses callback - her ses chunk'ı için çağrılır"""
        if self.is_recording:
            # Ses verisini numpy array'e çevir
            audio_chunk = np.frombuffer(in_data, dtype=np.int16)
            
            # VAD ile konuşma aktivitesini kontrol et
            if self._detect_speech_activity(audio_chunk):
                # Konuşma varsa buffer'a ekle
                self.current_speech_buffer.append(audio_chunk)
        
        return (in_data, pyaudio.paContinue)
    
    def get_transcription(self):
        """Bekleyen transkripsiyonları döndürür"""
        transcriptions = []
        while not self.transcription_queue.empty():
            transcriptions.append(self.transcription_queue.get())
        return transcriptions
    
    def adjust_sensitivity(self, speech_threshold=None, silence_threshold=None, timeout=None):
        """VAD hassasiyetini ayarlar"""
        if speech_threshold:
            self.speech_threshold = speech_threshold
            print(f"🔊 Konuşma eşiği: {speech_threshold}")
        
        if silence_threshold:
            self.silence_threshold = silence_threshold
            print(f"🔇 Sessizlik eşiği: {silence_threshold}")
        
        if timeout:
            self.silence_timeout = timeout
            print(f"⏱️ Sessizlik timeout: {timeout}s")
    
    def stop_recording(self):
        """Ses kaydını durdurur"""
        if hasattr(self, 'stream') and self.stream.is_active():
            self.is_recording = False
            
            # Eğer konuşma devam ediyorsa son kez işle
            if self.is_speaking and self.current_speech_buffer:
                print("\n🔄 Son konuşma parçası işleniyor...")
                self._process_speech()
            
            self.stream.stop_stream()
            self.stream.close()
            print("🛑 Kayıt durduruldu")
    
    def cleanup(self):
        """Kaynakları temizler"""
        self.stop_recording()
        self.audio.terminate()
        print("✅ Sistem temizlendi")

def main():
    """Ana fonksiyon"""
    print("=== VAD + Whisper Real-time STT ===\n")
    
    # STT sistemini başlat
    stt = VADWhisperSTT(
        model_size="base",    # "tiny", "base", "small", "medium", "large"
        language="tr",        # Türkçe
        device="cpu"          # GPU varsa "cuda"
    )
    
    try:
        # Kayıt başlat
        stt.start_recording()
        
        print("\n📢 Komutlar:")
        print("- Konuşun, sistem otomatik olarak başlangıç/bitiş algılar")
        print("- 's' = Hassasiyet ayarları")
        print("- 'quit' = Çıkış")
        print("- Ctrl+C = Hızlı çıkış\n")
        
        # Ana döngü
        while True:
            try:
                # Kullanıcı komutu al (timeout ile)
                print("\nKomut girin (veya Enter ile devam): ", end="", flush=True)
                
                # Basit input ile kontrol
                user_input = input().strip().lower()
                
                if user_input == 'quit':
                    break
                elif user_input == 's':
                    # Hassasiyet ayarları
                    print("\n🔧 Mevcut ayarlar:")
                    print(f"Konuşma eşiği: {stt.speech_threshold}")
                    print(f"Sessizlik eşiği: {stt.silence_threshold}")
                    print(f"Timeout süresi: {stt.silence_timeout}s")
                    
                    try:
                        new_speech = int(input("Yeni konuşma eşiği (şu an " + str(stt.speech_threshold) + "): ") or stt.speech_threshold)
                        new_silence = int(input("Yeni sessizlik eşiği (şu an " + str(stt.silence_threshold) + "): ") or stt.silence_threshold)
                        new_timeout = float(input("Yeni timeout süresi (şu an " + str(stt.silence_timeout) + "): ") or stt.silence_timeout)
                        
                        stt.adjust_sensitivity(new_speech, new_silence, new_timeout)
                    except ValueError:
                        print("⚠️ Geçersiz değer")
                
                # Yeni transkripsiyonları göster
                transcriptions = stt.get_transcription()
                for timestamp, text in transcriptions:
                    print(f"\n📝 [{timestamp}] {text}\n")
                
            except EOFError:
                break
            except KeyboardInterrupt:
                break
    
    except KeyboardInterrupt:
        print("\n🛑 Kullanıcı tarafından durduruldu")
    
    finally:
        stt.cleanup()
        print("👋 Program sonlandırıldı")

if __name__ == "__main__":
    main()