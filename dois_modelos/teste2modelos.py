import cv2
from ultralytics import YOLO

# Carregar o modelo pré-treinado YOLOv8
model_detection = YOLO("yolo11m-seg.pt")

# Carregar o modelo YOLOv8 treinado personalizado
model_classification = YOLO("modelo_treinado_v11_SGD-v4.pt")

# Inicializar a captura de vídeo
cap = cv2.VideoCapture("cam2.mp4")

# Inicializar o gravador de vídeo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_teste2911-3.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Set para armazenar IDs únicos detectados
unique_ids = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detecção inicial usando o modelo pré-treinado (YOLOv8n-seg)
    detection_results = model_detection.track(frame, imgsz=1280, conf=0.5, iou=0.5, persist=True)

    # Filtrar resultados relevantes para o modelo personalizado (exemplo: somente pessoas)
    filtered_detections = []
    for det in detection_results[0].boxes:
        if det.cls == 0:  # Classe '0' normalmente é 'person' nos modelos YOLO
            filtered_detections.append(det)

    # Refinar os resultados com o modelo treinado (aplicação personalizada)
    if filtered_detections:
        tracking_results = model_classification.track(frame, imgsz=1280, conf=0.85, persist=True, iou=0.5)
        
        # Extrair IDs dos rastreamentos
        if tracking_results[0].boxes.id is not None:
            tracks = tracking_results[0].boxes.id.cpu().numpy()
            unique_ids.update(tracks)

        # Atualizar a contagem total
        total_count = len(unique_ids)

        # Capturar o frame processado pelo YOLO
        annotated_frame = tracking_results[0].plot()  # Plota as detecções no frame original
    else:
        annotated_frame = frame

    # Adicionar a contagem total ao frame anotado
    cv2.putText(annotated_frame, f'Contagem: {total_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Exibir o frame com a contagem e salvar no vídeo
    cv2.imshow("Rastreamento", annotated_frame)
    out.write(annotated_frame)

    # Finalizar ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()

# Exibir contagem final
print(f"Contagem total de IDs únicos: {total_count}")
