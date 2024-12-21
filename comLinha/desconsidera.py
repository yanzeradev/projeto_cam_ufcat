import cv2
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
import torch
import torchreid

# Carregar modelos YOLO e Re-ID
model_detection = YOLO(r"C:\Users\yanka\Documents\DEV\YanProjeto\projeto_cam_ufcat\modelo_treinado_v11_SGD-v4.pt")
model_reid = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=751,  # O Market1501 possui 751 classes
    pretrained=True
).eval()

# Baixar e carregar o conjunto de dados Market1501 se necessário
datamanager = torchreid.data.ImageDataManager(
    root=r'C:\Users\yanka\Documents\DEV\YanProjeto\data',
    sources='market1501',
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=100,
    transforms=['random_flip', 'random_crop', 'color_jitter']
)

# Transformação para as imagens do Re-ID
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),  # Dimensão usada pelo modelo
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Função para extrair embeddings
def extract_features(bbox_image, model_reid):
    input_image = transform(bbox_image).unsqueeze(0)
    with torch.no_grad():
        features = model_reid(input_image).squeeze().cpu().numpy()
    return features

# Função para calcular similaridade entre embeddings
def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

# Aqui segue o restante do código com suas funcionalidades de detecção, rastreamento e Re-ID


# Inicializar a captura de vídeo
cap = cv2.VideoCapture(r"C:\Users\yanka\Documents\DEV\YanProjeto\projeto_cam_ufcat\cam2.mp4")

# Configurar dimensões e gravação
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("output_reid.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

# Variáveis globais
line_points = []
line_drawn = False
button_coords = (50, 100, 150, 150)
known_embeddings = {}
unique_id = 0
inside_count = 0
outside_count = 0
unique_ids_inside = set()
unique_ids_outside = set()

# Funções para interface
def draw_interface(frame):
    for i in range(1, len(line_points)):
        cv2.line(frame, line_points[i-1], line_points[i], (0, 255, 0), 2)
    cv2.rectangle(frame, (button_coords[0], button_coords[1]), (button_coords[2], button_coords[3]), (0, 0, 255), -1)
    cv2.putText(frame, "Salvar", (button_coords[0] + 10, button_coords[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def draw_line(event, x, y, flags, param):
    global line_points, line_drawn
    if event == cv2.EVENT_LBUTTONDOWN:
        if button_coords[0] < x < button_coords[2] and button_coords[1] < y < button_coords[3]:
            if len(line_points) > 1:
                line_drawn = True
                cv2.destroyWindow("Selecione a linha")
        else:
            line_points.append((x, y))

# Seleção da linha
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

# Função para verificar se um ponto cruzou a linha
def crossed_line(point, line_points):
    for i in range(1, len(line_points)):
        p1, p2 = line_points[i-1], line_points[i]
        d = np.cross(np.subtract(p2, p1), np.subtract(point, p1))
        if d < 0:
            return False
    return True

# Loop principal
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detecção com YOLO
    results = model_detection.track(frame, conf=0.6, imgsz=1280, persist=True, iou=0.5, tracker="bytetrack.yaml")

    for det in results[0].boxes:
        x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())
        bbox_image = frame[y1:y2, x1:x2]

        if bbox_image.size == 0:
            continue

        # Extrair embeddings usando o modelo Re-ID
        embedding = extract_features(bbox_image, model_reid)

        # Verificar Re-ID
        match_found = False
        for person_id, stored_embedding in known_embeddings.items():
            similarity = cosine_similarity(embedding, stored_embedding)
            if similarity > 0.7:
                match_found = True
                obj_id = person_id
                break

        if not match_found:
            unique_id += 1
            obj_id = unique_id
            known_embeddings[obj_id] = embedding

        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Verificar cruzamento de linha
        if obj_id not in unique_ids_inside and obj_id not in unique_ids_outside:
            if crossed_line(center, line_points):
                inside_count += 1
                unique_ids_inside.add(obj_id)
            else:
                outside_count += 1
                unique_ids_outside.add(obj_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {obj_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Desenhar a linha e contadores
    for i in range(1, len(line_points)):
        cv2.line(frame, line_points[i-1], line_points[i], (0, 255, 0), 2)
    cv2.putText(frame, f'Entraram: {inside_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Fora: {outside_count}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    out.write(frame)
    cv2.imshow("Rastreamento com Re-ID", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
