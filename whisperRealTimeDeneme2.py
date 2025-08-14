import argparse
import collections
import os
import queue
import sys
import time
from typing import Deque, Iterable, List, Tuple

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


# (Emniyet) Olası GPU görünürlüğünü kapatalım
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def list_input_devices():
    print("\n[Giriş Cihazları]")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"{i:2d}: {d['name']}  (default_sr={int(d['default_samplerate'])})")
    print()


class EnergyVADSegmenter:
    """
    RMS (enerji) tabanlı, ortama uyarlanabilir sessizlik/konuşma algılayıcı.
    Sessizlik eşiği aşılınca cümleyi bitirir.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        pre_speech_ms: int = 300,
        min_silence_ms: int = 600,
        min_utterance_ms: int = 350,
        max_utterance_ms: int = 30000,
        start_margin_db: float = 10.0,
        end_margin_db: float = 6.0,
        noise_floor_init_db: float = -55.0,
        noise_ema_alpha: float = 0.05,
        clamp_floor_db: float = -80.0,
    ):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.bytes_per_sample = 2
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * self.bytes_per_sample

        self.padding_frames = int(pre_speech_ms / frame_ms)
        self.silence_frames_to_end = int(min_silence_ms / frame_ms)
        self.min_utt_frames = int(max(1, min_utterance_ms / frame_ms))
        self.max_utt_frames = int(max(1, max_utterance_ms / frame_ms))

        self.start_margin_db = start_margin_db
        self.end_margin_db = end_margin_db

        self.noise_floor_db = noise_floor_init_db
        self.noise_ema_alpha = noise_ema_alpha
        self.clamp_floor_db = clamp_floor_db

        self.ring: Deque[bytes] = collections.deque(maxlen=self.padding_frames)
        self.triggered = False
        self.frames: List[bytes] = []
        self.silence_run = 0
        self.total_frames_in_utt = 0

    @staticmethod
    def dbfs_from_pcm16(frame_bytes: bytes) -> float:
        if not frame_bytes:
            return -np.inf
        x = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if x.size == 0:
            return -np.inf
        rms = np.sqrt(np.mean(x * x) + 1e-12)
        db = 20.0 * np.log10(rms + 1e-12)
        return float(db)

    def _update_noise_floor(self, db: float, is_speech_like: bool):
        if not is_speech_like and np.isfinite(db):
            db = max(self.clamp_floor_db, min(db, -10.0))
            self.noise_floor_db = (
                self.noise_ema_alpha * db + (1.0 - self.noise_ema_alpha) * self.noise_floor_db
            )

    def push(self, frame_bytes: bytes) -> Tuple[bool, bytes]:
        if len(frame_bytes) != self.frame_bytes:
            return False, b""

        db = self.dbfs_from_pcm16(frame_bytes)
        start_thresh = self.noise_floor_db + self.start_margin_db
        end_thresh = self.noise_floor_db + self.end_margin_db
        is_voice = db > start_thresh if not self.triggered else db > end_thresh

        if not self.triggered:
            self.ring.append(frame_bytes)
            if is_voice:
                self.triggered = True
                self.frames = list(self.ring)
                self.silence_run = 0
                self.total_frames_in_utt = len(self.frames)
            else:
                self._update_noise_floor(db, is_speech_like=False)
        else:
            self.frames.append(frame_bytes)
            self.total_frames_in_utt += 1

            if is_voice:
                self.silence_run = 0
            else:
                self.silence_run += 1
                self._update_noise_floor(db, is_speech_like=False)

            if self.total_frames_in_utt >= self.max_utt_frames:
                utter = b"".join(self.frames)
                self._reset_state()
                if self.total_frames_in_utt >= self.min_utt_frames:
                    return True, utter
                return False, b""

            if self.silence_run >= self.silence_frames_to_end:
                if self.total_frames_in_utt >= self.min_utt_frames:
                    utter = b"".join(self.frames)
                    self._reset_state()
                    return True, utter
                self._reset_state()
                return False, b""

        return False, b""

    def _reset_state(self):
        self.triggered = False
        self.frames = []
        self.ring.clear()
        self.silence_run = 0
        self.total_frames_in_utt = 0


def microphone_stream(device_index: int, sample_rate: int, frame_ms: int) -> Iterable[bytes]:
    blocksize = int(sample_rate * frame_ms / 1000)
    q_frames: queue.Queue = queue.Queue(maxsize=100)

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[stream status] {status}", file=sys.stderr)
        q_frames.put(bytes(indata))

    try:
        with sd.RawInputStream(
            samplerate=sample_rate,
            blocksize=blocksize,
            dtype="int16",
            channels=1,
            callback=callback,
            device=device_index if device_index is not None else None,
        ):
            while True:
                data = q_frames.get()
                yield data
    except Exception as e:
        print(f"[HATA] Mikrofon akışı başlatılamadı: {e}")
        print("Lütfen giriş cihazı index'ini doğru verdiğinden emin ol veya --list-devices ile kontrol et.")
        sys.exit(1)


class Transcriber:
    def __init__(self, model_name: str, language: str = "tr", compute_type: str = "int8"):
        print(f"[Whisper] CPU modunda model yükleniyor: {model_name} (compute_type={compute_type})")
        t0 = time.time()
        # CPU + int8
        self.model = WhisperModel(model_name, device="cpu", compute_type=compute_type)
        print(f"[Whisper] Hazır (yükleme: {time.time()-t0:.1f} sn)")
        self.language = language

    def transcribe_pcm16(self, pcm_bytes: bytes, sample_rate: int = 16000) -> str:
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _info = self.model.transcribe(
            audio=audio,
            language=self.language,
            beam_size=1,       # hız için greedy
            vad_filter=False,  # kendi VAD'ımız var
            word_timestamps=False,
        )
        return "".join(seg.text for seg in segments).strip()


def main():
    parser = argparse.ArgumentParser(description="Gerçek zamanlı Türkçe STT (CPU, enerji tabanlı sessizlik algılamalı)")
    parser.add_argument("--list-devices", action="store_true", help="Giriş cihazlarını listele ve çık")
    parser.add_argument("--device", type=int, default=None, help="Kullanılacak mikrofon cihaz index'i")
    parser.add_argument("--sr", type=int, default=16000, help="Örnekleme frekansı (16 kHz önerilir)")
    parser.add_argument("--frame-ms", type=int, default=30, help="Frame süresi (ms)")

    parser.add_argument("--model", type=str, default="small", help="faster-whisper modeli: tiny/base/small/medium/large-v3")
    parser.add_argument("--compute-type", type=str, default="int8", help="CPU compute type: int8 | int8_float16 | int16 | float32")

    # VAD ayarları
    parser.add_argument("--pre-speech", type=int, default=300, help="Cümle başına eklenecek tampon (ms)")
    parser.add_argument("--min-silence", type=int, default=600, help="Cümle bitimi için minimum sessizlik (ms)")
    parser.add_argument("--min-utt", type=int, default=350, help="Cümle sayılacak minimum uzunluk (ms)")
    parser.add_argument("--max-utt", type=int, default=30000, help="Tek cümle için azami süre (ms)")
    parser.add_argument("--start-margin", type=float, default=10.0, help="Başlangıç için gürültü tabanına ek dB")
    parser.add_argument("--end-margin", type=float, default=6.0, help="Devam/bitiş için gürültü tabanına ek dB")
    parser.add_argument("--noise-init", type=float, default=-55.0, help="Başlangıç gürültü tabanı (dBFS)")
    parser.add_argument("--noise-ema", type=float, default=0.05, help="Gürültü tabanı EMA katsayısı (0-1)")
    args = parser.parse_args()

    if args.list_devices:
        list_input_devices()
        sys.exit(0)

    transcriber = Transcriber(
        model_name=args.model,
        language="tr",
        compute_type=args.compute_type,
    )

    segmenter = EnergyVADSegmenter(
        sample_rate=args.sr,
        frame_ms=args.frame_ms,
        pre_speech_ms=args.pre_speech,
        min_silence_ms=args.min_silence,
        min_utterance_ms=args.min_utt,
        max_utterance_ms=args.max_utt,
        start_margin_db=args.start_margin,
        end_margin_db=args.end_margin,
        noise_floor_init_db=args.noise_init,
        noise_ema_alpha=args.noise_ema,
    )

    print("\n[Bilgi] CPU modunda çalışıyor. Konuşmaya başlayabilirsin. Sessizlik olduğunda cümleler ayrılacak.")
    print("[Bilgi] Çıkmak için Ctrl+C\n")

    sentence_idx = 1
    try:
        for frame in microphone_stream(device_index=args.device, sample_rate=args.sr, frame_ms=args.frame_ms):
            done, utter = segmenter.push(frame)
            if done and utter:
                text = transcriber.transcribe_pcm16(utter, sample_rate=args.sr).strip()
                if text and not text.endswith((".", "!", "?", "…", ".”", "!”", "?”")):
                    text += "."
                if text:
                    print(f"{sentence_idx}. Cümle= {text}")
                    sentence_idx += 1
    except KeyboardInterrupt:
        print("\n[Kapatılıyor] Görüşmek üzere!")
    except Exception as e:
        print(f"[HATA] {e}")


if __name__ == "__main__":
    main()
