import streamlit as st
import os
import tempfile
from pathlib import Path

# Imports locaux
from config import *
from utils.detection import load_models, train_team_classifier
from utils.visualization import generate_tracking_video, generate_radar_video, generate_voronoi_video
from sports.configs.soccer import SoccerPitchConfiguration

# ============================================
# CONFIGURATION DE LA PAGE
# ============================================
st.set_page_config(
    page_title="Football AI Analysis Platform",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CSS PERSONNALISÉ - THÈME FOOTBALL
# ============================================
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Variables de couleur - Thème Football */
    :root {
        --football-green: #00A651;
        --football-dark-green: #006B3D;
        --football-blue: #1E88E5;
        --football-light-blue: #42A5F5;
        --pitch-green: #7CB342;
        --white: #FFFFFF;
        --off-white: #F8F9FA;
        --dark-bg: #1A1D23;
        --card-bg: #252A34;
        --text-primary: #FFFFFF;
        --text-secondary: #B0BEC5;
        --border-color: rgba(255, 255, 255, 0.1);
        --success: #00C853;
        --warning: #FFB300;
        --error: #D32F2F;
    }
    
    /* Reset Streamlit Defaults */
    .main {
        background: linear-gradient(135deg, #1A1D23 0%, #252A34 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Header Principal */
    .main-header {
        background: linear-gradient(135deg, var(--football-dark-green) 0%, var(--football-green) 100%);
        padding: 2rem 3rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 166, 81, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: var(--white);
        margin: 0;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .main-subtitle {
        font-size: 1.1rem;
        font-weight: 400;
        color: rgba(255, 255, 255, 0.85);
        margin-top: 0.5rem;
        letter-spacing: 0.3px;
    }
    
    /* Divider avec effet terrain */
    .pitch-divider {
        height: 3px;
        background: linear-gradient(90deg, 
            transparent 0%, 
            var(--football-green) 25%, 
            var(--football-blue) 75%, 
            transparent 100%);
        margin: 2rem 0;
        border-radius: 2px;
    }
    
    /* Cards Info */
    .info-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .info-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, var(--football-green), var(--football-blue));
    }
    
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 166, 81, 0.15);
        border-color: var(--football-green);
    }
    
    .info-card-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--white);
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .info-card-text {
        font-size: 0.95rem;
        color: var(--text-secondary);
        line-height: 1.6;
    }
    
    /* Sidebar Styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: var(--dark-bg);
        border-right: 1px solid var(--border-color);
    }
    
    .sidebar-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--white);
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid var(--football-green);
    }
    
    /* Boutons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, var(--football-green) 0%, var(--football-dark-green) 100%);
        color: var(--white);
        font-weight: 600;
        font-size: 1.05rem;
        border: none;
        border-radius: 10px;
        padding: 0.85rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(0, 166, 81, 0.3);
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(0, 166, 81, 0.45);
        background: linear-gradient(135deg, var(--football-dark-green) 0%, var(--football-green) 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Download Buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--football-blue) 0%, var(--football-light-blue) 100%);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: var(--white);
        font-weight: 500;
        padding: 0.7rem 1.2rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(30, 136, 229, 0.4);
    }
    
    /* File Uploader */
    .uploadedFile {
        background: var(--card-bg);
        border: 2px dashed var(--football-green);
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Checkbox */
    .stCheckbox {
        color: var(--text-primary);
    }
    
    .stCheckbox > label {
        font-weight: 500;
        color: var(--text-primary);
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: var(--football-green);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--football-green), var(--football-blue));
    }
    
    /* Success/Warning/Error Messages */
    .stSuccess {
        background: rgba(0, 200, 83, 0.1);
        border-left: 4px solid var(--success);
        color: var(--text-primary);
        padding: 1rem;
        border-radius: 8px;
    }
    
    .stWarning {
        background: rgba(255, 179, 0, 0.1);
        border-left: 4px solid var(--warning);
        color: var(--text-primary);
        padding: 1rem;
        border-radius: 8px;
    }
    
    .stError {
        background: rgba(211, 47, 47, 0.1);
        border-left: 4px solid var(--error);
        color: var(--text-primary);
        padding: 1rem;
        border-radius: 8px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--card-bg);
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: 1px solid var(--border-color);
        color: var(--text-secondary);
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--football-green), var(--football-blue));
        color: var(--white);
        border-color: transparent;
    }
    
    /* Video Player */
    video {
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-right: 0.5rem;
    }
    
    .status-success {
        background: rgba(0, 200, 83, 0.15);
        color: var(--success);
        border: 1px solid var(--success);
    }
    
    .status-processing {
        background: rgba(255, 179, 0, 0.15);
        color: var(--warning);
        border: 1px solid var(--warning);
    }
    
    /* Footer */
    .app-footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 4rem;
        border-top: 1px solid var(--border-color);
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
    
    .app-footer a {
        color: var(--football-green);
        text-decoration: none;
        font-weight: 500;
        transition: color 0.3s ease;
    }
    
    .app-footer a:hover {
        color: var(--football-blue);
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        .main-subtitle {
            font-size: 0.95rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER PRINCIPAL
# ============================================
st.markdown("""
<div class="main-header">
    <h1 class="main-title">Football AI Analysis Platform</h1>
    <p class="main-subtitle">Analyse tactique automatisée par vision par ordinateur</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR - CONFIGURATION
# ============================================
with st.sidebar:
    st.markdown('<p class="sidebar-header">Configuration</p>', unsafe_allow_html=True)
    
    # Upload vidéo
    st.markdown("#### Téléchargement Vidéo")
    uploaded_video = st.file_uploader(
        "Sélectionner une vidéo de match",
        type=['mp4', 'avi', 'mov'],
        help="Formats supportés : MP4, AVI, MOV"
    )
    
    st.markdown('<div class="pitch-divider"></div>', unsafe_allow_html=True)
    
    # Paramètres
    st.markdown("#### Paramètres de Détection")
    confidence_threshold = st.slider(
        "Seuil de confiance",
        min_value=0.1,
        max_value=0.9,
        value=DEFAULT_CONFIDENCE,
        step=0.05,
        help="Score minimum pour qu'une détection soit acceptée"
    )
    
    st.markdown('<div class="pitch-divider"></div>', unsafe_allow_html=True)
    
    # Options de visualisation
    st.markdown("#### Vidéos à Générer")
    generate_tracking = st.checkbox("Détection & Tracking", value=True)
    generate_radar = st.checkbox("Vue Radar", value=True)
    generate_voronoi = st.checkbox("Diagramme de Voronoï", value=True)
    
    st.markdown('<div class="pitch-divider"></div>', unsafe_allow_html=True)
    
    # Info technique
    st.markdown("""
    <div style="background: rgba(0, 166, 81, 0.1); padding: 1rem; border-radius: 8px; border-left: 3px solid var(--football-green);">
        <p style="margin: 0; font-size: 0.85rem; color: var(--text-secondary);">
            <strong style="color: var(--white);">Note:</strong> Plus le seuil de confiance est élevé, 
            moins il y aura de détections, mais elles seront plus fiables.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# ZONE PRINCIPALE
# ============================================
if uploaded_video is None:
    # Page d'accueil - Présentation des fonctionnalités
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">
                <span style="color: var(--football-green); font-size: 1.5rem;">▸</span>
                Détection
            </div>
            <p class="info-card-text">
                Détection automatique des joueurs, gardiens, arbitres et ballon avec YOLOv11
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">
                <span style="color: var(--football-blue); font-size: 1.5rem;">▸</span>
                Analyse
            </div>
            <p class="info-card-text">
                Classification des équipes et suivi spatio-temporel avec ByteTrack
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">
                <span style="color: var(--pitch-green); font-size: 1.5rem;">▸</span>
                Visualisation
            </div>
            <p class="info-card-text">
                Vue radar, diagrammes de Voronoï et analyses tactiques avancées
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="pitch-divider"></div>', unsafe_allow_html=True)
    
    # Instructions
    st.markdown("""
    <div style="background: var(--card-bg); padding: 2rem; border-radius: 12px; border: 1px solid var(--border-color); margin-top: 2rem;">
        <h3 style="color: var(--white); margin-bottom: 1rem;">Comment utiliser la plateforme</h3>
        <ol style="color: var(--text-secondary); line-height: 2;">
            <li>Téléchargez une vidéo de match dans la barre latérale</li>
            <li>Ajustez le seuil de confiance selon vos besoins</li>
            <li>Sélectionnez les types de visualisations souhaités</li>
            <li>Lancez l'analyse et téléchargez les résultats</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

else:
    # Vidéo uploadée - Affichage du statut
    st.markdown(f"""
    <div style="background: rgba(0, 200, 83, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid var(--success); margin-bottom: 2rem;">
        <span class="status-badge status-success">VIDÉO CHARGÉE</span>
        <strong style="color: var(--white);">{uploaded_video.name}</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Bouton de génération
    if st.button("LANCER L'ANALYSE", type="primary"):
        
        # Sauvegarder la vidéo temporairement
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name
        
        # Barre de progression globale
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 1. Chargement des modèles
            status_text.markdown('<span class="status-badge status-processing">CHARGEMENT</span> Initialisation des modèles...', unsafe_allow_html=True)
            progress_bar.progress(10)
            
            detection_model, keypoint_model = load_models(
                YOLO11_MODEL_PATH,
                FIELD_DETECTION_MODEL_ID,
                ROBOFLOW_API_KEY
            )
            
            # 2. Entraînement du classificateur d'équipes
            status_text.markdown('<span class="status-badge status-processing">ENTRAÎNEMENT</span> Classification des équipes...', unsafe_allow_html=True)
            progress_bar.progress(20)
            
            team_classifier = train_team_classifier(
                video_path,
                detection_model,
                player_id=PLAYER_ID,
                stride=30
            )
            
            config = SoccerPitchConfiguration()
            
            output_paths = {}
            current_progress = 30
            
            # 3. Génération des vidéos
            if generate_tracking:
                status_text.markdown('<span class="status-badge status-processing">TRAITEMENT</span> Génération : Détection & Tracking...', unsafe_allow_html=True)
                output_path = os.path.join(OUTPUT_DIR, f"tracking_{uploaded_video.name}")
                generate_tracking_video(
                    video_path,
                    output_path,
                    detection_model,
                    team_classifier,
                    confidence=confidence_threshold
                )
                output_paths['Tracking'] = output_path
                current_progress += 20
                progress_bar.progress(current_progress)
            
            if generate_radar:
                status_text.markdown('<span class="status-badge status-processing">TRAITEMENT</span> Génération : Vue Radar...', unsafe_allow_html=True)
                output_path = os.path.join(OUTPUT_DIR, f"radar_{uploaded_video.name}")
                generate_radar_video(
                    video_path,
                    output_path,
                    detection_model,
                    keypoint_model,
                    team_classifier,
                    config,
                    confidence=confidence_threshold
                )
                output_paths['Vue Radar'] = output_path
                current_progress += 20
                progress_bar.progress(current_progress)
            
            if generate_voronoi:
                status_text.markdown('<span class="status-badge status-processing">TRAITEMENT</span> Génération : Diagramme de Voronoï...', unsafe_allow_html=True)
                output_path = os.path.join(OUTPUT_DIR, f"voronoi_{uploaded_video.name}")
                generate_voronoi_video(
                    video_path,
                    output_path,
                    detection_model,
                    keypoint_model,
                    team_classifier,
                    config,
                    confidence=confidence_threshold
                )
                output_paths['Voronoï'] = output_path
                current_progress += 20
                progress_bar.progress(current_progress)
            
            # 4. Terminé
            progress_bar.progress(100)
            status_text.markdown('<span class="status-badge status-success">TERMINÉ</span> Analyse complétée avec succès', unsafe_allow_html=True)
            
            st.markdown('<div class="pitch-divider"></div>', unsafe_allow_html=True)
            
            # Affichage des résultats
            st.success("Toutes les vidéos ont été générées avec succès")
            
            st.markdown("### Télécharger les Résultats")
            
            cols = st.columns(len(output_paths))
            
            for i, (name, path) in enumerate(output_paths.items()):
                with cols[i]:
                    with open(path, 'rb') as file:
                        st.download_button(
                            label=f"{name}",
                            data=file,
                            file_name=f"{name.lower().replace(' ', '_')}_{uploaded_video.name}",
                            mime="video/mp4"
                        )
            
            # Prévisualisation
            st.markdown('<div class="pitch-divider"></div>', unsafe_allow_html=True)
            st.markdown("### Prévisualisation des Vidéos")
            
            tabs = st.tabs(list(output_paths.keys()))
            
            for i, (name, path) in enumerate(output_paths.items()):
                with tabs[i]:
                    st.video(path)
        
        except Exception as e:
            st.error(f"Erreur lors de l'analyse : {str(e)}")
            st.exception(e)
        
        finally:
            # Nettoyage
            if os.path.exists(video_path):
                os.remove(video_path)

# ============================================
# FOOTER
# ============================================
st.markdown("""
<div class="app-footer">
    <p>
        Développé par <strong>Nada Alaoui</strong> | 
        <a href="https://intellcap.org" target="_blank">INTELLCAP</a> 2024-2025
    </p>
    <p style="font-size: 0.85rem; margin-top: 0.5rem;">
        Propulsé par YOLOv11, ByteTrack, SigLIP & Streamlit
    </p>
</div>
""", unsafe_allow_html=True)