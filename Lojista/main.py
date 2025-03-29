import cv2
from models import load_models
from utils import cap, out
from tracking import process_frame

# Carregar modelos e tracker
model_detection, model_classification, tracker = load_models()

# Inicializar contadores
inside_count = 0
outside_count = 0
unique_ids_inside = set()
unique_ids_outside = set()
classes = {0: "homem", 1: "mulher", 2: "naoidentificado"}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Processar frame
    inside_count, outside_count, unique_ids_inside, unique_ids_outside = process_frame(
        frame, model_detection, model_classification, tracker,
        inside_count, outside_count, unique_ids_inside, unique_ids_outside, classes
    )

    # Salvar e exibir frame
    out.write(frame)
    cv2.imshow("Rastreamento", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()
