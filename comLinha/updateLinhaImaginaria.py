import cv2
import numpy as np
from ultralytics import YOLO

# Carregar os modelos YOLO
model_detection = YOLO("yolo11n-seg.pt")
model_classification = YOLO("modelo_treinado_v11_SGD-v4.pt")

# Inicializar a captura de vídeo
cap = cv2.VideoCapture("cam2.mp4")

# Variáveis globais
line_points = []
line_drawn = False
button_coords = (50, 400, 150, 450)  # Coordenadas do "botão" (x1, y1, x2, y2)

# Função para desenhar pontos e botão
def draw_interface(frame):
    # Desenhar a linha conforme o usuário clica nos pontos
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
            # Botão "Salvar" foi clicado
            if len(line_points) > 1:
                line_drawn = True
                cv2.destroyWindow("Selecione a linha")
        else:
            # Adicionar ponto à linha
            line_points.append((x, y))

# Capturar o primeiro frame
ret, frame = cap.read()
frame_copy = frame.copy()

# Configurar a interface
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
    exit()

# Função para verificar se um ponto cruzou a linha poligonal
def crossed_line(point, line_points):
    for i in range(1, len(line_points)):
        p1, p2 = line_points[i-1], line_points[i]
        d = np.cross(np.subtract(p2, p1), np.subtract(point, p1))
        if d < 0:
            return False
    return True

# Inicializar contadores
inside_count = 0
outside_count = 0
unique_ids_inside = set()
unique_ids_outside = set()

# Definir as classes do modelo (ajuste conforme suas classes)
classes = {0: "homem", 1: "mulher", 2: "naoidentificado"}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detecção inicial usando YOLO
    results = model_detection.track(frame, imgsz=1280, conf=0.6, iou=0.7, persist=True)
    detections = [det for det in results[0].boxes if det.cls == 0]  # 'cls' = 0 é pessoa

    # Processar as predições do modelo personalizado
    tracking_results = model_classification.track(frame, imgsz=1280, conf=0.85, persist=True, iou=0.6)
    
    for det in tracking_results[0].boxes:
        x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
        obj_id = int(det.id)
        confidence = det.conf[0].item()  # Confiança
        class_id = int(det.cls[0].item())  # ID da classe
        class_name = classes.get(class_id, "desconhecido")  # Nome da classe
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Desenhar o BBox e a classe com a confiança e o ID
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {obj_id} {class_name} {confidence:.2f}', 
                    (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Verificar se a pessoa cruzou a linha
        if obj_id not in unique_ids_inside and obj_id not in unique_ids_outside:
            if crossed_line(center, line_points):
                # Se cruzou para dentro
                inside_count += 1
                unique_ids_inside.add(obj_id)  # Adiciona ao grupo dentro
                if obj_id in unique_ids_outside:
                    outside_count -= 1  # Decrementa no "fora" se a pessoa já estava fora
            else:
                # Se não cruzou, permanece fora
                outside_count += 1
                unique_ids_outside.add(obj_id)  # Adiciona ao grupo fora

    # Desenhar a linha salva no frame
    for i in range(1, len(line_points)):
        cv2.line(frame, line_points[i-1], line_points[i], (0, 255, 0), 2)

    # Exibir as contagens
    cv2.putText(frame, f'Entraram: {inside_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Fora: {outside_count}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Mostrar o frame com as BBoxes e contagens
    cv2.imshow("Rastreamento", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
