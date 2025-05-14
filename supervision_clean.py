import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

VIDEO_PATH = r"projeto_cam_ufcat\cam2.mp4"

# Carrega o vídeo e pega o primeiro frame para visualização
generator = sv.get_video_frames_generator(VIDEO_PATH)
iterator = iter(generator)
frame = next(iterator)

# Carrega o modelo YOLO
model = YOLO(r"projeto_cam_ufcat\modelo_genero_v12s_12-03-25_adam_imgz640-batch24_300epochs.pt")


# Cria o anotador de caixas
box_annotator = sv.BoxAnnotator(thickness=4, color=sv.Color.from_hex("#FFA500"))

# Função para processar cada frame do vídeo
def process_frame(frame: np.ndarray, i) -> np.ndarray:
    # Executa a detecção
    results = model(frame, imgsz=1280, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # Filtra detecções por confiança e classe (se necessário)
    detections = detections[detections.confidence > 0.7]
    # detections = detections[detections.class_id == 0]  # Descomente se quiser filtrar por classe
    
    # Anota as detecções no frame
    frame = box_annotator.annotate(
        scene=frame, 
        detections=detections
    )
    
    return frame

# Processa o vídeo completo
sv.process_video(
    source_path=VIDEO_PATH, 
    target_path="result.mp4", 
    callback=process_frame
)
