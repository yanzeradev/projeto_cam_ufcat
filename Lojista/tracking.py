import cv2
from scipy.spatial.distance import cosine
from collections import defaultdict, deque
import time
from utils import crossed_line, line_points
from models import load_models

# Carregar os modelos usando a função load_models
model_detection, model_classification, extractor = load_models()

# Configurações
SIMILARITY_THRESHOLD = 0.9  # Limiar de similaridade para re-identificação
FEATURE_TTL = 5400  # Tempo em segundos para manter as características no dicionário
FRAMES_TO_CONFIRM = 15  # Número de frames para confirmar a classe

def process_frame(frame, model_detection, model_classification, extractor, inside_count, outside_count, unique_ids_inside, unique_ids_outside, features_dict, classes, frame_counts, class_counts):
    # Dicionário para armazenar características com timestamp
    current_time = time.time()
    features_dict = {obj_id: (features, timestamp) for obj_id, (features, timestamp) in features_dict.items() if current_time - timestamp <= FEATURE_TTL}

    # Detecção inicial usando YOLO
    results = model_detection.track(frame, persist=True)
    detections = [det for det in results[0].boxes if det.cls == 0]  # 'cls' = 0 é pessoa

    tracking_results = model_classification.track(frame, conf=0.7, persist=True, iou=0.7)

    for det in tracking_results[0].boxes:
        x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())
        obj_id = int(det.id)
        confidence = det.conf[0].item()
        class_id = int(det.cls[0].item())
        class_name = classes.get(class_id, "desconhecido")
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Se o objeto não está em frame_counts, inicializar como "não identificado"
        if obj_id not in frame_counts:
            frame_counts[obj_id] = 0
            class_counts[obj_id] = defaultdict(int)
            class_id = 2  # Não identificado
            class_name = classes[2]

        # Incrementar o contador de frames para o objeto
        frame_counts[obj_id] += 1

        # Contar a classe atual
        class_counts[obj_id][class_id] += 1

        # Se atingiu 15 frames, determinar a classe predominante
        if frame_counts[obj_id] == FRAMES_TO_CONFIRM:
            predominant_class = max(class_counts[obj_id], key=class_counts[obj_id].get)
            class_id = predominant_class
            class_name = classes[predominant_class]
            print(f"Objeto {obj_id} confirmado como {class_name} após {FRAMES_TO_CONFIRM} frames.")

        # Se ainda não atingiu 15 frames, manter como "não identificado"
        elif frame_counts[obj_id] < FRAMES_TO_CONFIRM:
            class_id = 2  # Não identificado
            class_name = classes[2]

        # Extrair características utilizando o modelo OSNet
        person_crop = frame[int(y1):int(y2), int(x1):int(x2)]
        features = extractor(person_crop).detach().cpu().numpy().flatten()

        # Associar ID baseado em similaridade de características
        matched_id = None
        max_similarity = 0
        for known_id, (known_features, _) in features_dict.items():
            similarity = 1 - cosine(features, known_features)
            if similarity > max_similarity and similarity > SIMILARITY_THRESHOLD:
                max_similarity = similarity
                matched_id = known_id

        if matched_id:
            obj_id = matched_id
        else:
            features_dict[obj_id] = (features, current_time)

        # Atualizar contagem e IDs únicos
        if obj_id not in unique_ids_inside and obj_id not in unique_ids_outside:
            if crossed_line(center, line_points):
                inside_count += 1
                unique_ids_inside.add(obj_id)
            else:
                outside_count += 1
                unique_ids_outside.add(obj_id)

        if obj_id not in unique_ids_inside and obj_id in unique_ids_outside:
            if crossed_line(center, line_points):
                inside_count += 1
                unique_ids_inside.add(obj_id)
                outside_count -= 1
                unique_ids_outside.remove(obj_id)

        # Ajustar cor da caixa com base na classe
        if class_name == "homem":
            color = (0, 255, 0)
        elif class_name == "mulher":
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)

        # Desenhar BBox e a classe com confiança e ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'ID: {obj_id} {class_name} {confidence:.2f}', 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Desenhar a linha salva no frame
    for i in range(1, len(line_points)):
        cv2.line(frame, line_points[i-1], line_points[i], (0, 255, 0), 2)

    # Exibir as contagens
    cv2.putText(frame, f'Entraram: {inside_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Fora: {outside_count}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return inside_count, outside_count, unique_ids_inside, unique_ids_outside, features_dict, frame_counts, class_counts

# Inicialização das variáveis
frame_counts = defaultdict(int)
class_counts = defaultdict(lambda: defaultdict(int))

# Loop de processamento de vídeo
cap = cv2.VideoCapture('video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    inside_count, outside_count, unique_ids_inside, unique_ids_outside, features_dict, frame_counts, class_counts = process_frame(
        frame, model_detection, model_classification, extractor, inside_count, outside_count, unique_ids_inside, unique_ids_outside, features_dict, classes, frame_counts, class_counts
    )

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
