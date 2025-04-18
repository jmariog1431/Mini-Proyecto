import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Configuración de la cámara
cap = cv2.VideoCapture(0)

# Objeto virtual
obj_x, obj_y = 300, 300
obj_size = 50
obj_color = (0, 255, 0)  # Verde
holding = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Coordenadas del pulgar e índice
            thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            h, w, _ = frame.shape
            thumb_x, thumb_y = int(thumb.x * w), int(thumb.y * h)
            index_x, index_y = int(index.x * w), int(index.y * h)
            
            # Calcular distancia entre dedos
            distance = np.linalg.norm([thumb_x - index_x, thumb_y - index_y])
            
            # Dibujar línea entre dedos (feedback visual)
            cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 255), 2)
            
            # Lógica de agarre
            if distance < 40:  # Umbral de "pinch"
                if (obj_x - obj_size < index_x < obj_x + obj_size and 
                    obj_y - obj_size < index_y < obj_y + obj_size):
                    holding = True
            else:
                holding = False
            
            if holding:
                obj_x, obj_y = index_x, index_y
                obj_color = (255, 0, 0)  # Azul al agarrar
            else:
                obj_color = (0, 255, 0)   # Verde al soltar

    # Dibujar objeto
    cv2.rectangle(
        frame,
        (obj_x - obj_size, obj_y - obj_size),
        (obj_x + obj_size, obj_y + obj_size),
        obj_color,
        -1
    )

    # Instrucciones
    cv2.putText(frame, "Pinch to grab | 'Q' to quit", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Virtual Drag & Drop", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()