import cv2
import time
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils import crossed_line, line_points

# Configurações
SIMILARITY_THRESHOLD = 0.85  # Limiar de similaridade para re-identificação
FEATURE_TTL = 120  # Tempo em segundos para manter as características no dicionário

# Inicializar o DeepSORT
deepsort_tracker = DeepSort(max_age=50, n_init=3, nn_budget=100)

def process_frame(frame, model_detection, model_classification, extractor, inside_count, outside_count, unique_ids_inside, unique_ids_outside, features_dict, classes):
    # Dicionário para armazenar características com timestamp
    current_time = time.time()
    features_dict = {obj_id: (features, timestamp) for obj_id, (features, timestamp) in features_dict.items() if current_time - timestamp <= FEATURE_TTL}

    # Detecção inicial usando YOLO
    results = model_detection(frame, imgsz=1280, conf=0.5, iou=0.6)
    detections = [det for det in results[0].boxes if det.cls == 0]  # 'cls' = 0 é pessoa

    # Converter detecções para o formato do DeepSORT
    deepsort_detections = []
    for det in detections:
        x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())
        confidence = det.conf[0].item()
        class_id = int(det.cls[0].item())
        class_name = classes.get(class_id, "desconhecido")

        # Adicionar detecção ao DeepSORT
        deepsort_detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, class_name))

    # Rastrear objetos com DeepSORT
    tracks = deepsort_tracker.update_tracks(deepsort_detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        # Obter informações do track
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        class_name = track.get_det_class()
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Extrair características utilizando o modelo OSNet (opcional)
        person_crop = frame[int(y1):int(y2), int(x1):int(x2)]
        features = extractor(person_crop).detach().cpu().numpy().flatten()

        # Associar ID baseado em similaridade de características (opcional)
        matched_id = None
        max_similarity = 0
        for known_id, (known_features, _) in features_dict.items():
            similarity = 1 - cosine(features, known_features)
            if similarity > max_similarity and similarity > SIMILARITY_THRESHOLD:
                max_similarity = similarity
                matched_id = known_id

        if matched_id:
            track_id = matched_id
        else:
            features_dict[track_id] = (features, current_time)

        # Atualizar contagem e IDs únicos
        if track_id not in unique_ids_inside and track_id not in unique_ids_outside:
            if crossed_line(center, line_points):
                inside_count += 1
                unique_ids_inside.add(track_id)
            else:
                outside_count += 1
                unique_ids_outside.add(track_id)

        if track_id not in unique_ids_inside and track_id in unique_ids_outside:
            if crossed_line(center, line_points):
                inside_count += 1
                unique_ids_inside.add(track_id)
                outside_count -= 1
                unique_ids_outside.remove(track_id)

        # Ajustar cor da caixa com base na classe
        if class_name == "homem":
            color = (0, 255, 0)
        elif class_name == "mulher":
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)

        # Desenhar BBox e a classe com confiança e ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'ID: {track_id} {class_name}', 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Desenhar a linha salva no frame
    for i in range(1, len(line_points)):
        cv2.line(frame, line_points[i-1], line_points[i], (0, 255, 0), 2)

    # Exibir as contagens
    cv2.putText(frame, f'Entraram: {inside_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Fora: {outside_count}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return inside_count, outside_count, unique_ids_inside, unique_ids_outside, features_dict
