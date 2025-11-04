import cv2
import numpy as np
import pygame
import time
from collections import deque

class DrinkingDetector:
    def __init__(self, sound_file=None):
        # Pygame ses sistemi
        pygame.mixer.init()
        self.sound = None
        
        # Ses dosyasi varsa yuklemeye calis
        if sound_file:
            try:
                self.sound = pygame.mixer.Sound(sound_file)
                print(f"Ses dosyasi yuklendi: {sound_file}")
            except (pygame.error, FileNotFoundError):
                print(f"Ses dosyasi yuklenemedi: {sound_file}")
                print("   Ses olmadan devam ediliyor...")
        else:
            print("Ses dosyasi belirtilmedi, sessiz modda calisiyor.")
        
        # Yuz tanima
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Durumu takip etme
        self.is_drinking = False
        self.initial_white_pixels = None
        self.last_drink_time = time.time()
        self.MIN_DRINK_TIME = 2  # Minimum icme suresi (azaltildi)
        self.PIXEL_THRESHOLD = 1500  # Piksel degisiklik esigi (daha hassas yapildi)
        
        # Gurultuyu azaltmak icin gecmis degerleri saklama
        self.pixel_history = deque(maxlen=3)  # Daha kisa gecmis (hizli tepki)
        self.baseline_history = deque(maxlen=20)  # Baseline guncellemesi icin
        
        # Istatistikler
        self.drink_count = 0
        self.total_drink_time = 0
        
    def update_baseline(self, white_pixels):
        """Baseline degerini dinamik olarak guncelle"""
        if len(self.baseline_history) < 10:  # Daha hizli baseline hesaplama
            self.baseline_history.append(white_pixels)
            if self.initial_white_pixels is None and len(self.baseline_history) >= 5:  # Daha hizli baslangic
                self.initial_white_pixels = np.mean(self.baseline_history)
        else:
            # Sadece icmiyorken baseline'i guncelle
            if not self.is_drinking:
                self.baseline_history.append(white_pixels)
                self.initial_white_pixels = np.mean(self.baseline_history)
    
    def process_mouth_region(self, mouth_roi):
        """Agiz bolgesini isle ve beyaz piksel sayisini dondur"""
        if mouth_roi.size == 0:
            return 0, None
            
        # Gaussian blur ve threshold
        blurred = cv2.GaussianBlur(mouth_roi, (5, 5), 0)
        
        # Normal threshold daha iyi sonuc verebilir
        _, mouth_thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
        
        # Morfolojik islemler
        kernel = np.ones((3, 3), np.uint8)
        mouth_thresh = cv2.morphologyEx(mouth_thresh, cv2.MORPH_CLOSE, kernel)
        
        return np.sum(mouth_thresh == 255), mouth_thresh
    
    def detect_drinking(self, white_pixels):
        """Icme davranisini algila"""
        if self.initial_white_pixels is None or white_pixels == 0:
            return False
        
        # Gecmis degerleri kullanarak gurultuyu azalt
        self.pixel_history.append(white_pixels)
        avg_pixels = np.mean(self.pixel_history)
        
        # Baseline'dan sapma
        pixel_change = abs(avg_pixels - self.initial_white_pixels)
        
        current_time = time.time()
        
        # Daha hassas algilama
        if pixel_change > self.PIXEL_THRESHOLD:
            if not self.is_drinking:
                self.is_drinking = True
                self.last_drink_time = current_time
                self.drink_count += 1
                print(f"ICME ALGILANDI! (#{self.drink_count})")
                print(f"   Piksel degisimi: {pixel_change:.0f}")
                print(f"   Esik degeri: {self.PIXEL_THRESHOLD}")
                if self.sound:
                    self.sound.play()
                return True
            else:
                # Icme devam ediyor
                self.last_drink_time = current_time
        else:
            # Icme durdu mu kontrol et
            if self.is_drinking and (current_time - self.last_drink_time) > self.MIN_DRINK_TIME:
                self.is_drinking = False
                drink_duration = current_time - self.last_drink_time + self.MIN_DRINK_TIME
                self.total_drink_time += drink_duration
                print(f"ICME TAMAMLANDI! Sure: {drink_duration:.1f}s")
                return False
        
        return self.is_drinking
    
    def draw_info(self, frame, white_pixels, pixel_change=0):
        """Bilgi panelini ciz"""
        h, w = frame.shape[:2]
        
        # Bilgi paneli
        panel_height = 140
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Metin bilgileri
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 1
        
        # Baseline degerini guvenli sekilde hazirla
        baseline_value = "Hesaplaniyor..."
        if self.initial_white_pixels is not None:
            baseline_value = str(int(self.initial_white_pixels))
        
        # Durum rengi
        status_color = (0, 255, 0) if self.is_drinking else (255, 255, 255)
        
        # Metin listesi
        info_texts = [
            (f"Durum: {'ICIYOR' if self.is_drinking else 'BEKLIYOR'}", status_color),
            (f"Icme Sayisi: {self.drink_count}", color),
            (f"Toplam Sure: {self.total_drink_time:.1f}s", color),
            (f"Beyaz Piksel: {white_pixels}", color),
            (f"Baseline: {baseline_value}", color),
            (f"Degisim: {pixel_change:.0f}", color),
            (f"Esik: {self.PIXEL_THRESHOLD}", color)
        ]
        
        # Metinleri ekrana ciz
        for i, (text, text_color) in enumerate(info_texts):
            y_pos = 30 + i * 18
            cv2.putText(frame, text, (20, y_pos), font, font_scale, text_color, thickness)
        
        # Debug bilgisi - agiz bolgesinin durumu
        if pixel_change > self.PIXEL_THRESHOLD:
            cv2.putText(frame, ">>> HAREKET ALGILANDI! <<<", (20, panel_height - 10), 
                       font, 0.5, (0, 255, 255), 2)
    
    def run(self):
        """Ana dongu"""
        cap = cv2.VideoCapture(0)
        
        # Kamera ayarlari
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Kamera baslatildi. Cikmak icin 'q' tusuna basin.")
        print("Istatistikler icin 's' tusuna basin.")
        print("Hassasiyeti artirmak icin '+', azaltmak icin '-' basin.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Kamera goruntusu alinamadi!")
                    break
                
                # Ayna efekti icin goruntuyu ters cevir
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Yuz tespiti
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, 
                    minSize=(100, 100), maxSize=(400, 400)
                )
                
                white_pixels = 0
                pixel_change = 0
                
                for (x, y, w, h) in faces:
                    # Agiz bolgesi (yuzun alt %35'i - daha buyuk alan)
                    mouth_y_start = y + int(h * 0.65)
                    mouth_roi = gray[mouth_y_start:y + h, x:x + w]
                    
                    # Agiz bolgesini isle
                    white_pixels, mouth_thresh = self.process_mouth_region(mouth_roi)
                    
                    # Baseline guncelle
                    self.update_baseline(white_pixels)
                    
                    # Icme algila
                    if self.initial_white_pixels is not None:
                        pixel_change = abs(white_pixels - self.initial_white_pixels)
                        self.detect_drinking(white_pixels)
                    
                    # Gorsel cizimler
                    # Yuz cercevesi
                    face_color = (0, 255, 0) if self.is_drinking else (255, 0, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), face_color, 2)
                    
                    # Agiz bolgesi cercevesi
                    mouth_color = (0, 255, 255) if pixel_change > self.PIXEL_THRESHOLD else (255, 255, 0)
                    cv2.rectangle(frame, (x, mouth_y_start), 
                                (x + w, y + h), mouth_color, 2)
                    
                    # Agiz islenmis goruntusunu kucuk pencerede goster
                    if mouth_thresh is not None and mouth_thresh.size > 0:
                        mouth_resized = cv2.resize(mouth_thresh, (150, 80))
                        frame[150:230, frame.shape[1]-160:frame.shape[1]-10] = cv2.cvtColor(mouth_resized, cv2.COLOR_GRAY2BGR)
                
                # Bilgi panelini ciz
                self.draw_info(frame, white_pixels, pixel_change)
                
                # Gorüntuyu goster
                cv2.imshow('Icme Algilama Sistemi', frame)
                
                # Tus kontrolu
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    print(f"\nISTATISTIKLER:")
                    print(f"   Toplam icme sayisi: {self.drink_count}")
                    print(f"   Toplam icme suresi: {self.total_drink_time:.1f} saniye")
                    if self.drink_count > 0:
                        print(f"   Ortalama icme suresi: {self.total_drink_time/self.drink_count:.1f} saniye")
                    print(f"   Mevcut esik degeri: {self.PIXEL_THRESHOLD}")
                elif key == ord('r'):
                    # Reset
                    self.drink_count = 0
                    self.total_drink_time = 0
                    self.initial_white_pixels = None
                    self.pixel_history.clear()
                    self.baseline_history.clear()
                    print("Sistem sifirlandi!")
                elif key == ord('+') or key == ord('='):
                    # Hassasiyeti artir (esik degerini azalt)
                    self.PIXEL_THRESHOLD = max(500, self.PIXEL_THRESHOLD - 250)
                    print(f"Hassasiyet artirildi. Yeni esik: {self.PIXEL_THRESHOLD}")
                elif key == ord('-'):
                    # Hassasiyeti azalt (esik degerini artir)
                    self.PIXEL_THRESHOLD += 250
                    print(f"Hassasiyet azaltildi. Yeni esik: {self.PIXEL_THRESHOLD}")
        
        except KeyboardInterrupt:
            print("\nProgram kullanici tarafindan durduruldu.")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            pygame.quit()
            print("Program sonlandirildi.")

# Kullanim
if __name__ == "__main__":
    # Ses dosyasi istege bagli - varsa dosya yolunu belirtin
    # detector = DrinkingDetector("./ses.wav")  # Ses dosyasi ile
    detector = DrinkingDetector()  # Sessiz mod
    detector.run()
