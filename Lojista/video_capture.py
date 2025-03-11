import cv2
import ffmpeg
import numpy as np

# Configuração da câmera Intelbras
usuario = 'admin'
senha = 'senha'
ip = '127.0.0.1'
porta = '22479'
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

# Função para redimensionar e exibir a janela
def exibir_stream(window_name):
    while True:
        try:
            # Lê o frame do processo FFmpeg
            in_bytes = process.stdout.read(width * height * 3)
            if not in_bytes:
                break

            # Converte os bytes para um array numpy e em seguida para um frame OpenCV
            frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])

            # Obtém o tamanho atual da janela
            window_size = cv2.getWindowImageRect(window_name)
            current_width = window_size[2]
            current_height = window_size[3]

            # Redimensiona o frame para o tamanho da janela
            resized_frame = cv2.resize(frame, (current_width, current_height))

            # Exibe o frame redimensionado
            cv2.imshow(window_name, resized_frame)

            # Pressione 'q' para sair
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Erro: {e}")
            break

# Nome da janela
window_name = 'Camera-Intelbras'

# Cria a janela redimensionável
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Inicia a exibição da stream
exibir_stream(window_name)

# Finaliza o processo e libera recursos
process.wait()
cv2.destroyAllWindows()
