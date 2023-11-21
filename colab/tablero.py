import torch
import cv2
import numpy as np

# Función para detectar el tablero de ajedrez
def detectar_tablero(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Aplicar el detector de esquinas Shi-Tomasi
    esquinas = cv2.goodFeaturesToTrack(gris, 100, 0.01, 10)
 
    # Verificar si se encontraron esquinas
    if esquinas is not None:
        esquinas = np.int0(esquinas)
        return esquinas
    else:
        return None

# Cargar el modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/Adrian/Desktop/150epocas/yolov5/runs/train/exp2/weights/best.pt')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Detección del tablero
    esquinas_tablero = detectar_tablero(frame)

    if esquinas_tablero is not None:
        # Dibujar círculos en las esquinas del tablero
        for esquina in esquinas_tablero:
            x, y = esquina.ravel()
            cv2.circle(frame, (x, y), 3, 255, -1)

        # Detección de las piezas usando YOLOv5
        detect = model(frame)
        cv2.imshow('Detector de piezas', np.squeeze(detect.render()))

    t = cv2.waitKey(5)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()
