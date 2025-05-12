import cv2
import torch
import numpy as np
import supervision as sv
from ultralytics import YOLO

# Verificar se a GPU está disponível
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Carregar o modelo YOLO
model = YOLO(r"projeto_cam_ufcat\modelo_genero_v12s_12-03-25_adam_imgz640-batch24_300epochs.pt").to(device)

# Inicializar o rastreador ByteTrack da Supervision
tracker = sv.ByteTrack()

# Inicializar os anotadores da Supervision
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Inicializar o anotador de mapa de calor
heatmap_annotator = sv.HeatMapAnnotator(
    position=sv.Position.BOTTOM_CENTER,
    opacity=0.5,  # Opacidade do mapa de calor
    radius=10,    # Raio do círculo do mapa de calor
    kernel_size=25,
    top_hue=0,
    low_hue=125,
)

# Captura de vídeo
cap = cv2.VideoCapture(r"projeto_cam_ufcat\cam4.mp4")

# Configurações para salvar o vídeo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_path = r"output_cam.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para .mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Inicializar o gerador de frames de vídeo
frames_generator = sv.get_video_frames_generator(source_path=r"projeto_cam_ufcat\cam4.mp4", stride=1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detecção de objetos com YOLO
    results = model(frame, conf=0.5, iou=0.7)[0]

    # Converter os resultados para o formato Detections da Supervision
    detections = sv.Detections.from_ultralytics(results)

    # Atualizar o rastreador com as detecções
    detections = tracker.update_with_detections(detections)

    # Gerar labels com os IDs de rastreamento
    labels = [f"ID: {tracker_id}" for tracker_id in detections.tracker_id]

    # Gerar o mapa de calor
    annotated_frame = heatmap_annotator.annotate(scene=frame.copy(), detections=detections)

    # Anotar o frame com as caixas delimitadoras e labels
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    # Salvar o frame no vídeo de saída
    out.write(annotated_frame)

    # Mostrar o frame
    cv2.imshow("Frame", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()
