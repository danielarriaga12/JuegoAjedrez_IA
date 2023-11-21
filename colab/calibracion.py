import torch
import cv2
import numpy as np

# Carga del modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/Adrian/Desktop/150epocas/yolov5/runs/train/exp2/weights/best.pt')

# Captura de video desde la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detección de piezas con YOLOv5
    detect = model(frame)
    frame = np.squeeze(detect.render())

    # Filtrar el color blanco y el color rojo
    lower_white = np.array([200, 200, 200])
    upper_white = np.array([255, 255, 255])
    lower_red = np.array([0, 0, 100])
    upper_red = np.array([100, 100, 255])

    mask_white = cv2.inRange(frame, lower_white, upper_white)
    mask_red = cv2.inRange(frame, lower_red, upper_red)

    # Aplicar la binarización adaptativa con el método Gaussiano
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(
        gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 125, 1
    )

    # Combinar la máscara de blanco y rojo con la imagen binarizada
    result = cv2.bitwise_and(adaptive_thresh, cv2.bitwise_not(mask_white))
    result = cv2.bitwise_and(result, cv2.bitwise_not(mask_red))

    # Detectar líneas con la Transformada de Hough probabilística
    edges = cv2.Canny(result, threshold1=30, threshold2=100)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=10, maxLineGap=10
    )

    # Dibujar líneas en la imagen original
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Detectar cuadrados
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = []

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

        if len(approx) == 4:
            squares.append(approx)

    # Ordenar los cuadrados por su posición y tamaño
    squares = sorted(squares, key=lambda x: cv2.contourArea(x), reverse=True)[:64]

    # Dibujar cuadrados y etiquetas en la imagen original
    for i, square in enumerate(squares):
        x, y, w, h = cv2.boundingRect(square)
        letter = chr(65 + i % 8)  # A-H para las columnas
        number = str(8 - i // 8)  # 1-8 para las filas
        cv2.drawContours(frame, [square], -1, (0, 255, 0), 2)
        cv2.putText(frame, letter + number, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Muestra la imagen con piezas detectadas y cuadrados en tiempo real
    cv2.imshow("Chessboard Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()