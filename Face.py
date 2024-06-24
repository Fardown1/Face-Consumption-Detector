import cv2
import numpy as np
import pygame
import time

# Ses dosyasını başlatma
pygame.mixer.init()
sound = pygame.mixer.Sound("./mixkit-long-pop-2358.wav")

# Yüz tanıma sınıflandırıcısını yükleme
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Video kaynağını açma
cap = cv2.VideoCapture(0)

is_drinking = False
initial_white_pixels = None
last_drink_time = time.time()
MIN_DRINK_TIME = 5  # Minimum içme süresi (saniye)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Ağız bölgesinin daha doğru tespiti için ROI'yi ayarlama
        mouth_roi = gray[y + int(h * 0.65):y + h, x:x + w]
        blurred = cv2.GaussianBlur(mouth_roi, (7, 7), 0)
        _, mouth_thresh = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morfolojik işlemler
        kernel = np.ones((5, 5), np.uint8)
        mouth_thresh = cv2.morphologyEx(mouth_thresh, cv2.MORPH_CLOSE, kernel)
        white_pixels = np.sum(mouth_thresh == 255)

        if initial_white_pixels is None:
            initial_white_pixels = white_pixels

        # Beyaz piksel sayısı başlangıç değerinden farklıysa değişiklik algılanır
        if abs(white_pixels - initial_white_pixels) > 5000:  # Piksel değişiklik eşiği
            if not is_drinking:
                is_drinking = True
                last_drink_time = time.time()
                print("Tüketim algılandı!!")
                sound.play()
            else:
                # İçme devam ediyorsa, son içme zamanını güncelle
                last_drink_time = time.time()
        else:
            # İçme durduysa ve belirli bir süre geçtiyse
            if is_drinking and (time.time() - last_drink_time) > MIN_DRINK_TIME:
                is_drinking = False
                print("Tüketim tamamlandı!")

        # Ağız bölgesini ve yüzü çerçeveleme
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y + int(h * 0.65)), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
