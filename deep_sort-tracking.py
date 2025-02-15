from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch

# Verificar se a GPU está disponível
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Carregar o modelo YOLO
model_classification = YOLO(r"projeto_cam_ufcat\modelo_treinado_v8_Adam-310125-64batch.pt").to(device)

# Inicializar o DeepSort com ReID integrado
tracker = DeepSort(max_age=50, n_init=10, nms_max_overlap=0.5)

# Captura de vídeo
cap = cv2.VideoCapture(r"projeto_cam_ufcat\cam4.mp4")

# Configurações para salvar o vídeo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_path = r"output_cam.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para .mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detecção de objetos com YOLO
    results = model_classification.track(frame, conf=0.7, iou=0.7, persist=True)

    # Preparar as detecções para o DeepSort
    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], confidence, class_id))

    # Atualizar o rastreador DeepSort
    tracks = tracker.update_tracks(detections, frame=frame)

    # Desenhar os resultados
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Salvar o frame no vídeo de saída
    out.write(frame)

    # Mostrar o frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()
