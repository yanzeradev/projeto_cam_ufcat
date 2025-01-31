import cv2
import numpy as np

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(r"C:\Users\yanka\Documents\DEV\SenseVision\projeto_cam_ufcat\cam2.mp4")

# Obter dimensões e FPS do vídeo de entrada
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Inicializar o VideoWriter para salvar o vídeo processado
output_path = "output_cam.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec para .mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Variáveis globais
line_points = []
line_drawn = False
button_coords = (50, 100, 150, 150)  # Coordenadas do "botão" (x1, y1, x2, y2)

# Função para desenhar pontos e botão
def draw_interface(frame):
    for i in range(1, len(line_points)):
        cv2.line(frame, line_points[i-1], line_points[i], (0, 255, 0), 2)
    
    # Desenhar o botão "Salvar Linha"
    cv2.rectangle(frame, (button_coords[0], button_coords[1]), (button_coords[2], button_coords[3]), (0, 0, 255), -1)
    cv2.putText(frame, "Salvar", (button_coords[0] + 10, button_coords[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Função de callback para capturar cliques do mouse
def draw_line(event, x, y, flags, param):
    global line_points, line_drawn
    if event == cv2.EVENT_LBUTTONDOWN:
        if button_coords[0] < x < button_coords[2] and button_coords[1] < y < button_coords[3]:
            if len(line_points) > 1:
                line_drawn = True
                cv2.destroyWindow("Selecione a linha")
        else:
            line_points.append((x, y))

ret, frame = cap.read()
frame_copy = frame.copy()
cv2.imshow("Selecione a linha", frame_copy)
cv2.setMouseCallback("Selecione a linha", draw_line)

while not line_drawn:
    frame_copy = frame.copy()
    draw_interface(frame_copy)
    cv2.imshow("Selecione a linha", frame_copy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if not line_drawn or len(line_points) < 2:
    print("Linha não foi selecionada corretamente. Saindo.")
    cap.release()
    out.release()
    exit()

# Função para verificar se um ponto cruzou a linha poligonal
def crossed_line(point, line_points):
    for i in range(1, len(line_points)):
        p1, p2 = np.array(line_points[i-1]), np.array(line_points[i])
        d = np.cross(p2 - p1, point - p1)
        if d < 0:
            return False
    return True
