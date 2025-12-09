# ⚽ Football AI Analysis System

Système d'analyse vidéo de football utilisant la vision par ordinateur pour la détection, le tracking et l'analyse tactique.

## 🎯 Fonctionnalités

- **Détection d'objets** : Joueurs, gardiens, arbitres et ballon (YOLOv11)
- **Tracking** : Suivi des joueurs en temps réel (ByteTrack)
- **Classification d'équipes** : Attribution automatique via clustering non supervisé (SigLIP + UMAP + K-Means)
- **Visualisations tactiques** :
  - Vue radar (projection 2D)
  - Diagrammes de Voronoï (contrôle spatial)
  - Tracking avec annotations

## 🛠️ Technologies

- **YOLOv11** : Détection d'objets
- **YOLOv8x-pose** : Détection de points clés du terrain
- **ByteTrack** : Tracking multi-objets
- **SigLIP** : Embeddings visuels pour classification
- **Streamlit** : Interface web interactive

## 📦 Installation
```bash
# Cloner le repo
git clone https://github.com/ton-username/football-6ai.git
cd football-6ai

# Installer les dépendances
pip install -r requirements.txt

# Configurer les clés API
cp .env.example .env
# Éditer .env avec tes clés Roboflow
```

## 🚀 Utilisation
```bash
streamlit run app.py
```

## 📊 Résultats

### Détection (YOLOv11)
- **Joueurs** : 99.3% mAP50
- **Gardiens** : 94.1% mAP50
- **Arbitres** : 96.2% mAP50
- **Ballon** : 60.1% mAP50

### Keypoints terrain (YOLOv8x-pose)
- **mAP50 Box** : 99.5%
- **mAP50 Pose** : 97.0%

## 📄 Licence

MIT License - voir [LICENSE](LICENSE)

## 👤 Auteur

Nada - Stage INTELLCAP (Rabat) - 2025
