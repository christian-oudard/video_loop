import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)  # 0 = default camera
ret, img = cap.read()
cap.release()

plt.imshow(img)
