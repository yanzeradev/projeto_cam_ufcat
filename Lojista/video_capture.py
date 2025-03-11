import cv2
import ffmpeg
import numpy as np
import time
from datetime import datetime

# Configuração da câmera Intelbras
usuario = 'admin'
senha = 'senha'
ip = '127.0.0.1'
porta = '14389'
canal = '2'
subtipo = '0'

# URL RTSP da câmera Intelbras
url = f'rtsp://{usuario}:{senha}@{ip}:{porta}/cam/realmonitor?channel={canal}&subtype={subtipo}'

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
def salvar_video(duracao=20):
    # Nome do arquivo de saída com a data e hora atual
    data_atual = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    output_file = f'output_{data_atual}.mp4'
    
    # Inicializa o gravador de vídeo (usando o codec MJPEG)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Tempo final para salvar o vídeo
    tempo_final = time.time() + duracao

    while True:
        try:
            # Lê o frame do processo FFmpeg
            in_bytes = process.stdout.read(width * height * 3)
            if not in_bytes:
                break

            # Converte os bytes para um array numpy e em seguida para um frame OpenCV
            frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])

            # Salva o frame no arquivo de vídeo
            out.write(frame)

            # Exibe o frame
            cv2.imshow("Câmera Intelbras", frame)

            # Se o tempo de gravação tiver acabado, encerra a gravação
            if time.time() > tempo_final:
                break

            # Pressione 'q' para sair
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Erro: {e}")
            break

    # Libera os recursos
    out.release()
    cv2.destroyAllWindows()

# Inicia a captura e gravação de 10 segundos
salvar_video(duracao=20)

# Finaliza o processo de captura da câmera
process.wait()
