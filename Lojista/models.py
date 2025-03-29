from ultralytics import YOLO
import torch
import cv2
import logging
from deep_sort_realtime.deepsort_tracker import DeepSort

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_models():
    """
    Carrega os modelos YOLO para detecção e classificação, e o tracker DeepSORT.
    """
    # Verificar dispositivo (CPU ou GPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Usando dispositivo: {device}")

    # Carregar os modelos YOLO
    logger.info("Carregando modelo YOLO para detecção...")
    model_detection = YOLO("yolov8m.pt").to(device)
    
    logger.info("Carregando modelo YOLO para classificação...")
    model_classification = YOLO(r"projeto_cam_ufcat\modelo_treinado_v8_Adam-310125-64batch.pt").to(device)

    # Inicializar o tracker DeepSORT
    logger.info("Inicializando tracker DeepSORT...")
    tracker = DeepSort(
        max_age=50,            # Quantos frames um objeto é mantido antes de ser esquecido
        n_init=20,              # Frames necessários para confirmar um objeto
        nms_max_overlap=0.5,   # Usa GPU para extração de características
        nn_budget=100          # Memória do modelo de ReID
    )

    logger.info("Modelos carregados com sucesso!")
    return model_detection, model_classification, tracker
