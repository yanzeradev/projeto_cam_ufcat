from ultralytics import YOLO
import torchreid
import torch
import cv2

def load_models():
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
    return model_detection, model_classification, extractor