import cv2
from scipy.spatial.distance import cosine
from collections import defaultdict, deque
import time
import os
import numpy as np
from utils import crossed_line, line_points
from models import load_models

# Configurações
SIMILARITY_THRESHOLD = 0.8  # Ajuste o limiar de similaridade
FEATURE_TTL = 5400  # Tempo em segundos para manter as características
MAX_FEATURES_PER_ID = 20  # Número máximo de características armazenadas por ID
FRAMES_TO_CONFIRM = 15
SAVE_IMAGES = True  # Ativar/desativar salvamento de imagens
IMAGES_DIR = "id_images"  # Diretório para salvar as imagens por ID

# Criar diretório para salvar imagens, se não existir
if SAVE_IMAGES and not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

def process_frame(frame, model_detection, model_classification, extractor, inside_count, outside_count, unique_ids_inside, unique_ids_outside, features_dict, classes, frame_counts, class_counts, confirmed_ids):
    current_time = time.time()
    
    # Limpar características expiradas
    features_dict = {
        obj_id: (features_list, timestamp)
        for obj_id, (features_list, timestamp) in features_dict.items()
        if current_time - timestamp <= FEATURE_TTL
    }

    # Detecção inicial usando YOLO
    results = model_detection.track(frame, persist=True)
    detections = [det for det in results[0].boxes if det.cls == 0]  # 'cls' = 0 é pessoa

    tracking_results = model_classification.track(frame, conf=0.75, persist=True, iou=0.7)

    for det in tracking_results[0].boxes:
        x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())
        obj_id = int(det.id)
        confidence = det.conf[0].item()
        class_id = int(det.cls[0].item())
        class_name = classes.get(class_id, "desconhecido")
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Se o objeto já foi confirmado, usar o ID e a classe confirmados
        if obj_id in confirmed_ids:
            confirmed_data = confirmed_ids[obj_id]
            obj_id = confirmed_data["id"]
            class_id = confirmed_data["class_id"]
            class_name = confirmed_data["class_name"]
        else:
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

                # Armazenar o ID e a classe confirmados
                confirmed_ids[obj_id] = {
                    "id": obj_id,
                    "class_id": class_id,
                    "class_name": class_name
                }

                # Só atualizar a contagem após a confirmação do ID e da classe
                if obj_id not in unique_ids_inside and obj_id not in unique_ids_outside:
                    if crossed_line(center, line_points):
                        inside_count += 1
                        unique_ids_inside.add(obj_id)
                    else:
                        outside_count += 1
                        unique_ids_outside.add(obj_id)

            # Se ainda não atingiu 15 frames, manter como "não identificado"
            elif frame_counts[obj_id] < FRAMES_TO_CONFIRM:
                class_id = 2  # Não identificado
                class_name = classes[2]

        # Extrair características utilizando o modelo OSNet
        person_crop = frame[int(y1):int(y2), int(x1):int(x2)]
        features = extractor(person_crop).detach().cpu().numpy().flatten()

        # Armazenar múltiplas características para cada ID
        if obj_id not in features_dict:
            features_dict[obj_id] = (deque(maxlen=MAX_FEATURES_PER_ID), current_time)
        features_dict[obj_id][0].append(features)

        # Calcular a média das características para o ID atual
        avg_features_current = np.mean(features_dict[obj_id][0], axis=0)

        # Associar ID baseado em similaridade de características
        matched_id = None
        max_similarity = 0
        for known_id, (known_features_list, _) in features_dict.items():
            if known_id == obj_id:
                continue  # Ignorar o próprio objeto
            avg_features_known = np.mean(known_features_list, axis=0)
            similarity = 1 - cosine(avg_features_current, avg_features_known)
            if similarity > max_similarity and similarity > SIMILARITY_THRESHOLD:
                max_similarity = similarity
                matched_id = known_id

        if matched_id:
            print(f"Objeto {obj_id} associado ao ID {matched_id} com similaridade {max_similarity:.2f}")
            obj_id = matched_id

        # Salvar imagem do ID, se necessário
        if SAVE_IMAGES:
            id_dir = os.path.join(IMAGES_DIR, str(obj_id))
            if not os.path.exists(id_dir):
                os.makedirs(id_dir)
            image_path = os.path.join(id_dir, f"{int(current_time)}.jpg")
            cv2.imwrite(image_path, person_crop)

        # Atualizar contagem e IDs únicos (apenas para objetos confirmados)
        if obj_id in confirmed_ids:
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

    return inside_count, outside_count, unique_ids_inside, unique_ids_outside, features_dict, frame_counts, class_counts, confirmed_ids
