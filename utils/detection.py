import supervision as sv
import torch
from ultralytics import YOLO
from inference import get_model
from sports.common.team import TeamClassifier
from tqdm import tqdm
import numpy as np

def load_models(yolo_path, roboflow_model_id, roboflow_api_key):
    """
    Charge les modèles YOLO11 et le modèle de détection de points clés.
    """
    detection_model = YOLO(yolo_path)
    keypoint_model = get_model(model_id=roboflow_model_id, api_key=roboflow_api_key)
    return detection_model, keypoint_model

def train_team_classifier(video_path, detection_model, player_id=2, stride=30):
    """
    Entraîne le classificateur d'équipes sur la vidéo fournie.
    """
    frame_generator = sv.get_video_frames_generator(source_path=video_path, stride=stride)
    
    crops = []
    for frame in tqdm(frame_generator, desc='Collecting crops for team classification'):
        result = detection_model.predict(frame, conf=0.3)[0]
        detections = sv.Detections.from_ultralytics(result)
        players_detections = detections[detections.class_id == player_id]
        players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
        crops += players_crops
    
    team_classifier = TeamClassifier(device="cuda" if torch.cuda.is_available() else "cpu")
    team_classifier.fit(crops)
    
    return team_classifier