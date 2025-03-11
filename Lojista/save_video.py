import cv2
import ffmpeg
import numpy as np
import time
from datetime import datetime

# Configuração da câmera
usuario = 'admin'
senha = 'senha'
ip = '127.0.0.1'
porta = '19462'
canal = '2'
starttime = '2025_03_10_08_00_00'
endtime = '2025_03_10_08_00_30'

# URL RTSP da câmera
url = f'rtsp://{usuario}:{senha}@{ip}:{porta}/cam/playback?channel={canal}&starttime={starttime}&endtime={endtime}'

# Comando FFmpeg para acessar a câmera e converter para BGR24
process = (
    ffmpeg
    .input(url, rtsp_transport='tcp')  # Alterado para TCP para maior estabilidade
    .output('pipe:', format='rawvideo', pix_fmt='bgr24')
    .run_async(pipe_stdout=True)
)

# Tamanho do frame (ajustar conforme sua câmera)
width, height = 1280, 720  # Ajuste para a resolução correta
fps = 15  # FPS de captura

# Função para capturar e salvar o vídeo
def salvar_video():
    # Nome do arquivo de saída com a data e hora atual
    output_file = f'output-{starttime}-{endtime}.mp4'
    
    # Inicializa o gravador de vídeo (usando o codec MJPEG)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Tempo de início e fim em segundos
    start_time = time.time()
    end_time = start_time + (datetime.strptime(endtime, '%Y_%m_%d_%H_%M_%S') - datetime.strptime(starttime, '%Y_%m_%d_%H_%M_%S')).total_seconds()

    while time.time() < end_time:
        try:
            # Lê o frame do processo FFmpeg
            in_bytes = process.stdout.read(width * height * 3)
            if not in_bytes:
                break

            # Converte os bytes para um array numpy e em seguida para um frame OpenCV
            frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])

            # Salva o frame no arquivo de vídeo
            out.write(frame)

        except Exception as e:
            print(f"Erro: {e}")
            break

    # Libera os recursos
    out.release()
    process.terminate()
    process.wait()

# Inicia a captura e gravação
salvar_video()
