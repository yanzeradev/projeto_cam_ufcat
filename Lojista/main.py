import cv2
import pickle
from collections import defaultdict
from models import load_models
from utils import cap, out
from tracking import process_frame

# Carregar os modelos usando a função load_models
model_detection, model_classification, extractor = load_models()

# Inicializar contadores
inside_count = 0
outside_count = 0
unique_ids_inside = set()
unique_ids_outside = set()
features_dict = {}

# Inicializar contadores de frames e classes
frame_counts = defaultdict(int)  # Conta o número de frames por objeto
class_counts = defaultdict(lambda: defaultdict(int))  # Conta a frequência de classes por objeto

# Definir as classes do modelo
classes = {0: "homem", 1: "mulher", 2: "naoidentificado"}

confirmed_ids = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    inside_count, outside_count, unique_ids_inside, unique_ids_outside, features_dict, frame_counts, class_counts, confirmed_ids = process_frame(
        frame, model_detection, model_classification, extractor, inside_count, outside_count, unique_ids_inside, unique_ids_outside, features_dict, classes, frame_counts, class_counts, confirmed_ids
    )

    # Salvar o frame processado no arquivo de saída
    out.write(frame)

    # Mostrar o frame com as BBoxes e contagens
    cv2.imshow("Rastreamento", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Salvar características em disco
with open('features_dict.pkl', 'wb') as f:
    pickle.dump(features_dict, f)

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()
