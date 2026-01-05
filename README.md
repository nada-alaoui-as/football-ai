# Football AI Analysis System

Computer vision system for football video analysis, providing real-time detection, tracking, and tactical insights.

## Features

- **Object Detection**: Players, goalkeepers, referees, and ball detection using YOLOv11
- **Multi-Object Tracking**: Real-time player tracking with ByteTrack
- **Team Classification**: Unsupervised team assignment using SigLIP embeddings with UMAP dimensionality reduction and K-Means clustering
- **Tactical Visualizations**:
  - Radar view with 2D pitch projection
  - Voronoï diagrams for spatial control analysis
  - Annotated tracking with player IDs and team colors

## Technical Stack

- **YOLOv11**: Object detection (players, goalkeepers, referees, ball)
- **YOLOv8x-pose**: Field keypoint detection for homography transformation
- **ByteTrack**: Multi-object tracking algorithm
- **SigLIP**: Visual embeddings for team classification
- **UMAP + K-Means**: Unsupervised clustering for team assignment
- **Streamlit**: Interactive web application

## Installation
```bash
# Clone the repository
git clone https://github.com/nada-alaoui-as/football-ai.git
cd football-ai

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your Roboflow API key
```

## Usage
```bash
streamlit run app.py
```

Upload a football video and select the desired analysis type:
- Detection and tracking with team classification
- Radar view (2D tactical projection)
- Voronoï diagrams (spatial control analysis)

## Model Performance

### YOLOv11 - Object Detection
Trained on 450 images (100 epochs, batch size 2, image size 1280x1280)

| Class      | Instances | Precision | Recall | mAP50 |
|------------|-----------|-----------|--------|-------|
| Players    | 973       | 92.9%     | 98.9%  | 99.3% |
| Goalkeepers| 39        | 87.2%     | 87.2%  | 94.1% |
| Referees   | 117       | 81.9%     | 94.9%  | 96.2% |
| Ball       | 45        | 83.3%     | 55.5%  | 60.1% |

**Overall**: 87.4% mAP50, 59.3% mAP50-95

### YOLOv8x-pose - Field Keypoints
Trained on 222 images (100 epochs, 32 keypoints)

- **Box mAP50**: 99.5%
- **Pose mAP50**: 97.0%
- **Precision**: 99.8%
- **Recall**: 100%

## Project Structure
```
football-ai/
├── app.py                 # Streamlit application
├── config.py             # Configuration file
├── requirements.txt      # Python dependencies
├── utils/                # Core modules
│   ├── detection.py      # Object detection
│   ├── tracking.py       # Player tracking
│   ├── classification.py # Team assignment
│   └── visualization.py  # Rendering utilities
└── notebooks/            # Development notebooks
    └── pipeline.ipynb    # Complete analysis pipeline
```

## Key Technical Details

**Team Classification Pipeline**:
1. Extract player crops (stride 30 frames)
2. Generate 768-dimensional embeddings using SigLIP
3. Reduce to 3D using UMAP
4. Cluster into 2 teams using K-Means
5. Resolve goalkeeper assignments using spatial centroids

**Homography Transformation**:
- 32 field keypoints detected with YOLOv8x-pose
- ViewTransformer computes perspective transformation matrix
- Projects player positions onto 2D tactical view

**Optimization**:
- AdamW optimizer with decoupled weight decay
- Adaptive learning rate with momentum
- Regularization to prevent overfitting

## Limitations

- Ball detection performance limited by small object size and rapid movement
- Dataset size (450 images) below recommended threshold (500+)
- Lighting variations affect detection consistency
- Single-camera perspective limits full pitch coverage

## Future Improvements

- Multi-camera fusion for complete pitch coverage
- Enhanced ball tracking with temporal filtering
- Real-time performance optimization
- Advanced tactical metrics (possession zones, pressure maps, distance covered)
- Predictive models for player behavior

## License

MIT License - see [LICENSE](LICENSE)

## Author

Nada Alaoui - 2025

## Acknowledgments

Developed using:
- Roboflow for dataset management and annotation
- Ultralytics YOLO for detection models
- Supervision library for tracking and visualization
- Google Colab for model training (Tesla T4 GPU)
