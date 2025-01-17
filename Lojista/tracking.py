import cv2
from modules.utils import crossed_line, line_points

def process_frame(frame, model_detection, model_classification, extractor, inside_count, outside_count, unique_ids_inside, unique_ids_outside, features_dict, classes):
    # Detecção inicial usando YOLO
    results = model_detection.track(frame, imgsz=1280, conf=0.6, iou=0.7, persist=True)
    detections = [det for det in results[0].boxes if det.cls == 0]  # 'cls' = 0 é pessoa

    tracking_results = model_classification.track(frame, imgsz=1280, conf=0.85, persist=True, iou=0.6)

    for det in tracking_results[0].boxes:
        x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())
        obj_id = int(det.id)
        confidence = det.conf[0].item()
        class_id = int(det.cls[0].item())
        class_name = classes.get(class_id, "desconhecido")
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Extrair características utilizando o modelo OSNet
        person_crop = frame[int(y1):int(y2), int(x1):int(x2)]
        features = extractor(person_crop)
        features_dict[obj_id] = features

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

    return inside_count, outside_count, unique_ids_inside, unique_ids_outside, features_dict
