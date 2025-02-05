from ultralytics import YOLO
import torchreid
import torch
import cv2
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_models():
    """
    Carrega os modelos YOLO para detecção e classificação, e o extrator de características OSNet.
    """
    # Verificar dispositivo (CPU ou GPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Usando dispositivo: {device}")

    # Carregar os modelos YOLO
    logger.info("Carregando modelo YOLO para detecção...")
    model_detection = YOLO("yolov8m.pt").to(device)
    
    logger.info("Carregando modelo YOLO para classificação...")
    model_classification = YOLO(r"C:\Users\yanka\Documents\DEV\SenseVision\projeto_cam_ufcat\modelo_treinado_v8_Adam-310125-64batch.pt").to(device)

    # Inicializar o extrator de características do Torchreid (OSNet)
    logger.info("Carregando extrator de características OSNet...")
    extractor = torchreid.utils.FeatureExtractor(
        model_name='osnet_x1_0',
        model_path=r'C:\Users\yanka\Documents\DEV\SenseVision\projeto_cam_ufcat\osnet_x1_0_imagenet.pth',
        device=device
    )

    logger.info("Modelos carregados com sucesso!")
    return model_detection, model_classification, extractor
