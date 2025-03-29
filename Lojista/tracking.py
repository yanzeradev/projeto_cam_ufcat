import cv2
import time
import os
import numpy as np
from utils import crossed_line, line_points
from models import load_models

# Configurações
FRAMES_TO_CONFIRM = 20
SAVE_IMAGES = True
IMAGES_DIR = "id_images"
MAX_AGE = 40  # Tempo máximo (frames) que um tracker pode ficar sem atualização

def process_frame(frame, model_detection, model_classification, tracker, inside_count, outside_count, 
                 unique_ids_inside, unique_ids_outside, classes):
    
    # Detecção inicial usando YOLO
    results = model_classification.predict(frame, conf=0.5, iou=0.7)
    
    # Preparar detecções para o DeepSORT
    detections = []
    for det in results[0].boxes:
        x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())
        confidence = det.conf[0].item()
        class_id = int(det.cls[0].item())
        
        detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, class_id))

    # Atualizar tracker com as novas detecções
    tracks = tracker.update_tracks(detections, frame=frame)

    # Processar tracks atualizados
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue

        # Obter coordenadas da bounding box
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        obj_id = track.track_id
        class_id = track.get_det_class()
        class_name = classes.get(class_id, "desconhecido")

        # Atualizar contagem
        if obj_id not in unique_ids_inside and obj_id not in unique_ids_outside:
            if crossed_line(center, line_points):
                inside_count += 1
                unique_ids_inside.add(obj_id)
            else:
                outside_count += 1
                unique_ids_outside.add(obj_id)

        # Ajustar cor da caixa com base na classe
        color = (0, 255, 0) if class_name == "homem" else (255, 0, 0) if class_name == "mulher" else (0, 0, 255)

        # Desenhar elementos
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'ID: {obj_id} {class_name}', (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Salvar imagem do ID (opcional)
        if SAVE_IMAGES:
            save_id_image(frame, obj_id, x1, y1, x2, y2)

    # Desenhar a linha de contagem
    for i in range(1, len(line_points)):
        cv2.line(frame, line_points[i-1], line_points[i], (0, 255, 0), 2)

    # Exibir contagens
    cv2.putText(frame, f'Entraram: {inside_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Fora: {outside_count}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return inside_count, outside_count, unique_ids_inside, unique_ids_outside

def save_id_image(frame, obj_id, x1, y1, x2, y2):
    """Salva imagens dos IDs para verificação manual"""
    id_dir = os.path.join(IMAGES_DIR, str(obj_id))
    if not os.path.exists(id_dir):
        os.makedirs(id_dir)
    person_crop = frame[y1:y2, x1:x2]
    if person_crop.size > 0:
        cv2.imwrite(os.path.join(id_dir, f"{int(time.time())}.jpg"), person_crop)
