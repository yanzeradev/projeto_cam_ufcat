import traceback
import os
import zipfile
from ultralytics import YOLO

# URLs e caminhos
DATASET_URL = "https://app.roboflow.com/ds/jwBWxutnlY?key=O49p0wUgvK"
DATASET_ZIP_PATH = r"/datasets/jwBWxutnlY.zip"  # Caminho para salvar o dataset baixado
DATASET_DIR = r"/datasets/jwBWxutnlY"  # Caminho para descompactar o dataset
MODEL_NAME = "yolov8n.pt"  # Atualizado para YOLOv8
MODEL_SAVE_PATH = r"modelo_treinado_v8n_adam_140225.pt"

def download_and_unzip_dataset(dataset_url, zip_path, extract_dir):
    """Baixa e descompacta o dataset."""
    import requests

    # Cria o diretório de extração, se não existir
    os.makedirs(extract_dir, exist_ok=True)

    # Baixa o dataset
    print(f"Baixando dataset de {dataset_url}...")
    response = requests.get(dataset_url, stream=True)
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Dataset baixado e salvo em {zip_path}.")

    # Descompacta o dataset
    print(f"Descompactando dataset em {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Dataset descompactado em {extract_dir}.")

if __name__ == '__main__':
    try:
        # Baixa e descompacta o dataset
        download_and_unzip_dataset(DATASET_URL, DATASET_ZIP_PATH, DATASET_DIR)

        # Carregar o modelo YOLO
        print("Carregando modelo...")
        model = YOLO(MODEL_NAME)
        print("Modelo carregado com sucesso!")

        # Configuração do treinamento
        train_params = {
            'data': os.path.join(DATASET_DIR, "data.yaml"),  # Caminho local para o dataset
            "epochs": 100,                   # Número de épocas
            "batch": -1,                      # Tamanho do batch
            "lr0": 0.001,                     # Taxa de aprendizado inicial
            "lrf": 0.15,                      # Taxa de aprendizado final
            "momentum": 0.937,                # Momentum
            "imgsz": 960,                     # Tamanho da imagem de entrada
            "conf": 0.45,                     # Threshold de confiança
            "iou": 0.5,                       # Threshold de IOU para supressão não máxima
            "optimizer": "Adam",              # Otimizador: 'Adam' ou 'SGD'
            "patience": 15,
            "device": "0",  
            "val": True
        }

        # Treinar o modelo
        print("Iniciando treinamento...")
        model.train(**train_params)
        print("Treinamento concluído com sucesso!")

        # Avaliar o modelo
        print("Avaliando o modelo...")
        metrics = model.val()
        print("Métricas de avaliação:", metrics)

        # Salvar o modelo treinado
        print(f"Salvando modelo em {MODEL_SAVE_PATH}...")
        model.save(MODEL_SAVE_PATH)
        print(f"Modelo treinado salvo com sucesso em {MODEL_SAVE_PATH}")

        # Exportar o modelo treinado
        print("Exportando modelo para ONNX...")
        model.export(format="onnx")
        print("Modelo exportado com sucesso.")

    except Exception as e:
        print("Erro detectado durante a execução:")
        traceback.print_exc()
