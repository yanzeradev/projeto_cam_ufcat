import cv2
from ultralytics import YOLO

video_path = 'cam4.mp4'
model_crowdhuman = YOLO('modelo_crowdhuman_v12n_16-06-25_adam_imgz640-batch26_200epochs.pt')
model_genero = YOLO('modelo_genero_v12m_28-05-25_adam_imgz640-batch10_200epochs.pt')

cap = cv2.VideoCapture(video_path)
while cap:
    ok, frame = cap.read()
    if not ok:
        break

    results = model_crowdhuman.predict(frame, iou=0.6, conf=0.6, classes=[0, 1])
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if box.cls[0] == 0:
                    class_name = 'Cabeca'
                    color = (255, 0, 0)
                else:
                    class_name = 'Pessoa'
                    color = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {box.conf[0]:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    cv2.imshow('CrowdHuman Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

