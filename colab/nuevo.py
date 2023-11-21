import cv2
import sys

# Configuración de la cámara
cap_index = 0  # Puedes cambiar esto según tu configuración de cámara
cap_api = cv2.CAP_ANY  # API de captura predeterminada
cap = cv2.VideoCapture(cap_index, cap_api)

if not cap.isOpened():
    print("No se pudo abrir la cámara. Verifique la conexión de su cámara.")
    sys.exit(0)

# Dimensiones del tablero (esquinas internas)
board_dimensions = (7, 7)  # Para un tablero de ajedrez de 8x8

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al leer el cuadro. Verifique la conexión de su cámara.")
        continue

    # Convertir a escala de grises para la detección de esquinas
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    retval, corners = cv2.findChessboardCorners(gray, patternSize=board_dimensions)

    if retval:
        # Dibujar las esquinas detectadas en el frame
        frame = cv2.drawChessboardCorners(frame, board_dimensions, corners, retval)

    # Mostrar el frame
    cv2.imshow('Tablero de Ajedrez - Detección de Esquinas', frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()