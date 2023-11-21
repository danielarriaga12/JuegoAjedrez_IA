import torch
import cv2
import numpy as np  # Import NumPy

model = torch.hub.load('ultralytics/yolov5', 'custom', path='/content/drive/MyDrive/colab/best.pt')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    detect = model(frame)

    cv2.imshow('detector de piezas', np.squeeze(detect.render()))

    t = cv2.waitKey(5)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()