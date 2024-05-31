import cv2
import mediapipe as mp
import pygame
from datetime import datetime
import time

# Inicializa MediaPipe para la detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Inicializa pygame para la reproducción de sonido
pygame.init()
sound = pygame.mixer.Sound("sound.mp3")

# Inicializa la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Variable para contar las detecciones de manos
hand_count = 0
current_hour = datetime.now().hour

# Función para guardar el recuento en un archivo de texto
def save_count(hour, count):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    with open("hand_detection_count.txt", "a") as file:
        file.write(f"{timestamp} - {count} times\n")

# Variables para el delay
last_detection_time = 0
detection_delay = 2  # segundos

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convierte el frame de BGR a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesa el frame para la detección de manos
    result = hands.process(rgb_frame)

    # Si se detecta una mano y han pasado al menos 2 segundos desde la última detección
    current_time = time.time()
    if result.multi_hand_landmarks and (current_time - last_detection_time) > detection_delay:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Reproduce el sonido
            pygame.mixer.Sound.play(sound)
            # Incrementa el contador de detecciones de manos
            hand_count += 1
            # Actualiza el tiempo de la última detección
            last_detection_time = current_time

            # Verifica si la hora ha cambiado
            if datetime.now().hour != current_hour:
                # Guarda el recuento de la hora anterior
                save_count(current_hour, hand_count)
                # Resetea el contador y actualiza la hora actual
                hand_count = 0
                current_hour = datetime.now().hour

    # Muestra el frame
    cv2.imshow('Hand Detection', frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Guarda el recuento final antes de salir
save_count(current_hour, hand_count)

# Libera la cámara y destruye todas las ventanas
cap.release()
cv2.destroyAllWindows()
pygame.quit()
