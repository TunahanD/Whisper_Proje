# =============================
# FILE: memory_monitor.py
# =============================
"""
Memory monitoring modülü - Threshold tabanlı RAM izleme
- 8GB üzeri: Warning
- 10GB üzeri: Program restart
- Zero-impact monitoring (transkripsiyon sonrası check)
"""

import os
import sys
import psutil
from typing import Optional

class MemoryMonitor:
    def __init__(self, warning_threshold_gb: float = 3.0, restart_threshold_gb: float = 8.0):
        """
        Memory monitor initialization
        
        Args:
            warning_threshold_gb: GB cinsinden uyarı eşiği (default: 3GB)
            restart_threshold_gb: GB cinsinden restart eşiği (default: 8GB)
        """
        self.warning_threshold = warning_threshold_gb * 1024 * 1024 * 1024  # Bytes
        self.restart_threshold = restart_threshold_gb * 1024 * 1024 * 1024  # Bytes
        self.process = psutil.Process(os.getpid())
        self.warning_shown = False  # Sadece bir kez uyarı göster
        self.last_memory_mb = 0
        
        print(f"[Memory Monitor] Başlatıldı - Warning: {warning_threshold_gb}GB, Restart: {restart_threshold_gb}GB")
    
    def get_memory_usage(self) -> tuple[float, float]:
        """
        Mevcut memory kullanımını döndür
        
        Returns:
            tuple: (memory_bytes, memory_mb)
        """
        try:
            memory_info = self.process.memory_info()
            memory_bytes = memory_info.rss  # Resident Set Size (physical memory)
            memory_mb = memory_bytes / (1024 * 1024)
            return memory_bytes, memory_mb
        except Exception as e:
            print(f"[Memory Monitor] Hata - Memory bilgisi alınamadı: {e}")
            return 0, 0
    
    def check_memory_threshold(self) -> Optional[str]:
        """
        Memory threshold kontrolü yap
        
        Returns:
            str: 'warning', 'restart', veya None
        """
        memory_bytes, memory_mb = self.get_memory_usage()
        self.last_memory_mb = memory_mb
        
        if memory_bytes == 0:
            return None
        
        # Restart threshold check (önce bu kontrol edilmeli)
        if memory_bytes >= self.restart_threshold:
            memory_gb = memory_mb / 1024
            print(f"\n{'='*50}")
            print(f"🚨 KRİTİK: Memory kullanımı {memory_gb:.1f}GB!")
            print(f"🔄 Program yeniden başlatılıyor...")
            print(f"{'='*50}")
            return 'restart'
        
        # Warning threshold check
        elif memory_bytes >= self.warning_threshold and not self.warning_shown:
            memory_gb = memory_mb / 1024
            print(f"\n{'='*40}")
            print(f"⚠️  UYARI: Memory kullanımı yüksek!")
            print(f"📊 Mevcut kullanım: {memory_gb:.1f}GB")
            print(f"🔄 5GB üzeri otomatik restart")
            print(f"{'='*40}")
            self.warning_shown = True
            return 'warning'
        
        # Memory düştüyse warning flag'i reset et
        elif memory_bytes < self.warning_threshold and self.warning_shown:
            self.warning_shown = False
            memory_gb = memory_mb / 1024
            print(f"✅ Memory kullanımı normale döndü: {memory_gb:.1f}GB")
        
        return None
    
    def get_memory_stats(self) -> dict:
        """
        Detaylı memory istatistikleri
        
        Returns:
            dict: Memory usage statistics
        """
        memory_bytes, memory_mb = self.get_memory_usage()
        memory_gb = memory_mb / 1024
        
        # Sistem memory bilgisi
        system_memory = psutil.virtual_memory()
        total_system_gb = system_memory.total / (1024**3)
        available_system_gb = system_memory.available / (1024**3)
        system_usage_percent = system_memory.percent
        
        return {
            'process_memory_mb': memory_mb,
            'process_memory_gb': memory_gb,
            'warning_threshold_gb': self.warning_threshold / (1024**3),
            'restart_threshold_gb': self.restart_threshold / (1024**3),
            'system_total_gb': total_system_gb,
            'system_available_gb': available_system_gb,
            'system_usage_percent': system_usage_percent,
            'warning_active': self.warning_shown
        }
    
    def print_memory_stats(self):
        """
        Memory istatistiklerini yazdır (debug için)
        """
        stats = self.get_memory_stats()
        print(f"\n=== Memory Stats ===")
        print(f"Process: {stats['process_memory_gb']:.2f}GB")
        print(f"System: {stats['system_usage_percent']:.1f}% ({stats['system_available_gb']:.1f}GB available)")
        print(f"Thresholds: Warning={stats['warning_threshold_gb']:.1f}GB, Restart={stats['restart_threshold_gb']:.1f}GB")
        print(f"==================")
    
    def should_restart(self) -> bool:
        """
        Restart gerekip gerekmediğini kontrol et
        
        Returns:
            bool: True if restart needed
        """
        result = self.check_memory_threshold()
        return result == 'restart'
    
    def reset_warning_flag(self):
        """
        Warning flag'ini manuel olarak reset et
        """
        self.warning_shown = False
        print("[Memory Monitor] Warning flag resetlendi")


def restart_program():
    """
    Programı yeniden başlat (graceful restart)
    """
    print("[Restart] Program yeniden başlatılıyor...")
    print("[Restart] Mevcut işlemler tamamlanıyor...")
    
    try:
        # Mevcut script argumentlarını al
        args = sys.argv
        
        # Python executable path
        python_executable = sys.executable
        
        print(f"[Restart] Komut: {python_executable} {' '.join(args)}")
        print("[Restart] 3 saniye içinde yeniden başlatılacak...")
        
        import time
        time.sleep(3)
        
        # Yeni process başlat
        os.execv(python_executable, [python_executable] + args)
        
    except Exception as e:
        print(f"[Restart] HATA: Program yeniden başlatılamadı: {e}")
        print("[Restart] Manuel restart gerekiyor...")
        sys.exit(1)


# Test fonksiyonu (standalone çalıştırma için)
if __name__ == "__main__":
    monitor = MemoryMonitor(warning_threshold_gb=0.1, restart_threshold_gb=0.2)  # Test thresholds
    
    print("Memory monitor test...")
    monitor.print_memory_stats()
    
    result = monitor.check_memory_threshold()
    print(f"Test result: {result}")
    
    if result == 'restart':
        print("Restart simulation (test modunda gerçek restart yapılmaz)")