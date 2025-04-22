import cv2
import mediapipe as mp
import numpy as np

def dibujar_estrella(imagen, centro, tamaño, color):
    puntos = np.array([
        [0, -1], [0.2245, -0.309], [0.9511, -0.309], 
        [0.3633, 0.118], [0.5878, 0.809],
        [0, 0.382], [-0.5878, 0.809], 
        [-0.3633, 0.118], [-0.9511, -0.309],
        [-0.2245, -0.309]
    ]) * tamaño

    puntos = puntos.astype(np.int32) + centro
    cv2.polylines(imagen, [puntos], isClosed=True, color=color, thickness=2)
    cv2.fillPoly(imagen, [puntos], color=color)

manos_mp = mp.solutions.hands
dibujo_mp = mp.solutions.drawing_utils
detector_manos = manos_mp.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

camara = cv2.VideoCapture(0)

pos_x, pos_y = 300, 300
tamaño_estrella = 50
color_estrella = (0, 255, 0)
sujetar = False

while camara.isOpened():
    ret, fotograma = camara.read()
    if not ret:
        break

    fotograma = cv2.flip(fotograma, 1)
    fotograma_rgb = cv2.cvtColor(fotograma, cv2.COLOR_BGR2RGB)
    resultados = detector_manos.process(fotograma_rgb)

    if resultados.multi_hand_landmarks:
        for mano in resultados.multi_hand_landmarks:
            dibujo_mp.draw_landmarks(fotograma, mano, manos_mp.HAND_CONNECTIONS)
            
            pulgar = mano.landmark[manos_mp.HandLandmark.THUMB_TIP]
            indice = mano.landmark[manos_mp.HandLandmark.INDEX_FINGER_TIP]
            
            alto, ancho, _ = fotograma.shape
            pulgar_x, pulgar_y = int(pulgar.x * ancho), int(pulgar.y * alto)
            indice_x, indice_y = int(indice.x * ancho), int(indice.y * alto)
            
            distancia = np.linalg.norm([pulgar_x - indice_x, pulgar_y - indice_y])
            
            cv2.line(fotograma, (pulgar_x, pulgar_y), (indice_x, indice_y), (0, 255, 255), 2)
            
            if distancia < 40:
                if (pos_x - tamaño_estrella < indice_x < pos_x + tamaño_estrella and 
                    pos_y - tamaño_estrella < indice_y < pos_y + tamaño_estrella):
                    sujetar = True
            else:
                sujetar = False
            
            if sujetar:
                pos_x, pos_y = indice_x, indice_y
                color_estrella = (255, 0, 0)
            else:
                color_estrella = (0, 255, 0)

    dibujar_estrella(fotograma, (pos_x, pos_y), tamaño_estrella, color_estrella)

    cv2.putText(fotograma, "Haz 'pinch' para agarrar | 'Q' para salir", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Arrastrar y soltar virtual", fotograma)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camara.release()
cv2.destroyAllWindows()
