from ultralytics import YOLO
import cv2

model_classification = YOLO(r"C:\Users\yanka\Documents\DEV\SenseVision\projeto_cam_ufcat\modelo_treinado_v8_Adam-310125.pt")
cap = cv2.VideoCapture(r"C:\Users\yanka\Documents\DEV\SenseVision\projeto_cam_ufcat\cam2.mp4")

while(cap):
    ret, frame = cap.read()
    results = model_classification.track(frame, imgsz=1280, conf=0.55, iou=0.7, persist=True, show=True)
