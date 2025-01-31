from ultralytics import YOLO
import torchreid
import torch
import cv2

def load_models():
    # Carregar os modelos YOLO
    model_detection = YOLO("yolov8n-seg.pt")
    model_classification = YOLO(r"C:\Users\yanka\Documents\DEV\SenseVision\projeto_cam_ufcat\modelo_treinado_v8_Adam-310125-64batch.pt")

    # Inicializar o extrator de características do Torchreid
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = torchreid.utils.FeatureExtractor(
        model_name='osnet_x1_0',
        model_path=r'C:\Users\yanka\Documents\DEV\SenseVision\projeto_cam_ufcat\osnet_x1_0_imagenet.pth',
        device=device
    )
    return model_detection, model_classification, extractor
