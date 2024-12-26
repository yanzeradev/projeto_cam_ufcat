import cv2
import numpy as np
from ultralytics import YOLO
import torchreid
import torch
import pickle

# Carregar os modelos YOLO
model_detection = YOLO("yolov8n.pt")
model_classification = YOLO(r"C:\Users\yanka\Documents\DEV\YanProjeto\projeto_cam_ufcat\modelo_treinado_v11_SGD-v4.pt")

# Inicializar o extrator de características do Torchreid
device = 'cuda' if torch.cuda.is_available() else 'cpu'
extractor = torchreid.utils.FeatureExtractor(
    model_name='osnet_x1_0',
    model_path=r'C:\Users\yanka\Documents\DEV\YanProjeto\osnet_x1_0_imagenet.pth',
    device=device
)

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(r"C:\Users\yanka\Documents\DEV\YanProjeto\projeto_cam_ufcat\cam2.mp4")

# Obter dimensões e FPS do vídeo de entrada
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Inicializar o VideoWriter para salvar o vídeo processado
output_path = "output_cam3.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec para .mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Variáveis globais
line_points = []
line_drawn = False
button_coords = (50, 100, 150, 150)  # Coordenadas do "botão" (x1, y1, x2, y2)

# Função para desenhar pontos e botão
def draw_interface(frame):
    for i in range(1, len(line_points)):
        cv2.line(frame, line_points[i-1], line_points[i], (0, 255, 0), 2)
    
    # Desenhar o botão "Salvar Linha"
    cv2.rectangle(frame, (button_coords[0], button_coords[1]), (button_coords[2], button_coords[3]), (0, 0, 255), -1)
    cv2.putText(frame, "Salvar", (button_coords[0] + 10, button_coords[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Função de callback para capturar cliques do mouse
def draw_line(event, x, y, flags, param):
    global line_points, line_drawn
    if event == cv2.EVENT_LBUTTONDOWN:
        if button_coords[0] < x < button_coords[2] and button_coords[1] < y < button_coords[3]:
            if len(line_points) > 1:
                line_drawn = True
                cv2.destroyWindow("Selecione a linha")
        else:
            line_points.append((x, y))

ret, frame = cap.read()
frame_copy = frame.copy()
cv2.imshow("Selecione a linha", frame_copy)
cv2.setMouseCallback("Selecione a linha", draw_line)

while not line_drawn:
    frame_copy = frame.copy()
    draw_interface(frame_copy)
    cv2.imshow("Selecione a linha", frame_copy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if not line_drawn or len(line_points) < 2:
    print("Linha não foi selecionada corretamente. Saindo.")
    cap.release()
    out.release()
    exit()

# Função para verificar se um ponto cruzou a linha poligonal
def crossed_line(point, line_points):
    for i in range(1, len(line_points)):
        p1, p2 = np.array(line_points[i-1]), np.array(line_points[i])
        d = np.cross(p2 - p1, point - p1)
        if d < 0:
            return False
    return True

# Inicializar contadores
inside_count = 0
outside_count = 0
unique_ids_inside = set()
unique_ids_outside = set()
features_dict = {}

# Definir as classes do modelo
classes = {0: "homem", 1: "mulher", 2: "naoidentificado"}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

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
    
    # Salvar o frame processado no arquivo de saída
    out.write(frame)

    # Mostrar o frame com as BBoxes e contagens
    cv2.imshow("Rastreamento", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Salvar características em disco
with open('features_dict.pkl', 'wb') as f:
    pickle.dump(features_dict, f)

cap.release()
out.release()
cv2.destroyAllWindows()
