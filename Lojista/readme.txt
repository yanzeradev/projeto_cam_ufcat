projeto/
├── main.py                # Arquivo principal para execução
├── config.py              # Configurações gerais (caminhos, parâmetros, etc.)
├── modules/
│   ├── detection.py       # Funções relacionadas à detecção (YOLO)
│   ├── classification.py  # Funções relacionadas à classificação (TorchReid e YOLO)
│   ├── tracking.py        # Funções relacionadas ao rastreamento e cruzamento de linha
│   ├── visualization.py   # Funções de interface e visualização
│   └── utils.py           # Funções auxiliares (gerais)
└── outputs/
    ├── processed_videos/  # Vídeos processados
    ├── logs/              # Logs da aplicação
    └── features/          # Dados salvos (ex.: features_dict.pkl)
