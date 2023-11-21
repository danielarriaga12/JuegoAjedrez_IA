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
        esquinas = np.float32(esquinas)
        return esquinas
    else:
        return None

# Cargar el modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/Adrian/Desktop/150epocas/yolov5/runs/train/exp2/weights/best.pt')

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    # Detección del tablero
    esquinas_tablero = detectar_tablero(frame)

    if esquinas_tablero is not None and len(esquinas_tablero) == 4:
        # Definir los puntos de la cuadrícula rectangular ideal del tablero
        cuadricula_rectangular = np.array([[0, 0], [400, 0], [400, 400], [0, 400]], dtype=np.float32)
        
        # Encontrar la homografía
        h, _ = cv2.findHomography(esquinas_tablero, cuadricula_rectangular)

        # Aplicar la homografía para corregir la perspectiva
        imagen_corregida = cv2.warpPerspective(frame, h, (400, 400))

        # Dibujar los cuadrados fijos en la imagen corregida
        for fila in range(8):
            for columna in range(8):
                x1, y1 = (columna * 50, fila * 50)
                x2, y2 = ((columna + 1) * 50, (fila + 1) * 50)
                cv2.rectangle(imagen_corregida, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Detección de las piezas usando YOLOv5 en la imagen corregida
        detect = model(imagen_corregida)
        cv2.imshow('Detector de piezas', np.squeeze(detect.render()))

    t = cv2.waitKey(5)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()
