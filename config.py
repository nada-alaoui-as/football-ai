import os
from dotenv import load_dotenv

# Charger variables d'environnement
load_dotenv()

# API Keys
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

# Modèles
YOLO11_MODEL_PATH = "models/yolov11_best.pt"
FIELD_DETECTION_MODEL_ID = "football-field-detection-f07vi/14"

# Paramètres détection
DEFAULT_CONFIDENCE = 0.30
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

# Couleurs
COLORS = {
    'team1': '#00BFFF',  # Bleu
    'team2': '#FF1493',  # Rose
    'referee': '#FFD700'  # Or
}

# Output
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)