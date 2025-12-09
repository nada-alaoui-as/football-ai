import supervision as sv
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer
from tqdm import tqdm
import numpy as np
from .helpers import resolve_goalkeepers_team_id

CONFIG = SoccerPitchConfiguration()

def generate_tracking_video(
    source_video_path, 
    target_video_path, 
    detection_model, 
    team_classifier,
    confidence=0.3
):
    """
    Génère une vidéo avec détection, tracking et annotation des joueurs/ballon.
    """
    BALL_ID = 0
    GOALKEEPER_ID = 1
    PLAYER_ID = 2
    REFEREE_ID = 3

    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        thickness=2
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        text_color=sv.Color.from_hex('#000000'),
        text_position=sv.Position.BOTTOM_CENTER
    )
    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex('#FFD700'),
        base=25,
        height=21,
        outline_thickness=1
    )

    tracker = sv.ByteTrack()
    tracker.reset()

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    video_sink = sv.VideoSink(target_video_path, video_info=video_info)
    frame_generator = sv.get_video_frames_generator(source_video_path)

    with video_sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames, desc="Generating tracking video"):
            result = detection_model.predict(frame, conf=confidence)[0]
            detections = sv.Detections.from_ultralytics(result)

            ball_detections = detections[detections.class_id == BALL_ID]
            ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

            all_detections = detections[detections.class_id != BALL_ID]
            all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
            all_detections = tracker.update_with_detections(detections=all_detections)

            goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
            players_detections = all_detections[all_detections.class_id == PLAYER_ID]
            referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

            players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
            players_detections.class_id = team_classifier.predict(players_crops)
            goalkeepers_detections.class_id = resolve_goalkeepers_team_id(players_detections, goalkeepers_detections)
            referees_detections.class_id -= 1

            all_detections = sv.Detections.merge([players_detections, goalkeepers_detections, referees_detections])

            labels = [f"#{int(tid)}" for tid in all_detections.tracker_id]
            all_detections.class_id = all_detections.class_id.astype(int)

            annotated_frame = frame.copy()
            annotated_frame = ellipse_annotator.annotate(scene=annotated_frame, detections=all_detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=all_detections, labels=labels)
            annotated_frame = triangle_annotator.annotate(scene=annotated_frame, detections=ball_detections)

            video_sink.write_frame(annotated_frame)


def generate_radar_video(
    source_video_path,
    target_video_path,
    detection_model,
    keypoint_model,
    team_classifier,
    config,
    confidence=0.3
):
    """
    Génère une vidéo vue radar avec projection des joueurs sur le terrain.
    """
    BALL_ID = 0
    GOALKEEPER_ID = 1
    PLAYER_ID = 2
    REFEREE_ID = 3

    tracker = sv.ByteTrack()
    tracker.reset()

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    video_sink = sv.VideoSink(target_video_path, video_info=video_info)
    frame_generator = sv.get_video_frames_generator(source_video_path)

    with video_sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames, desc="Generating radar video"):
            result = detection_model.predict(frame, conf=confidence)[0]
            detections = sv.Detections.from_ultralytics(result)

            ball_detections = detections[detections.class_id == BALL_ID]
            ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

            all_detections = detections[detections.class_id != BALL_ID]
            all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
            all_detections = tracker.update_with_detections(all_detections)

            goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
            players_detections = all_detections[all_detections.class_id == PLAYER_ID]
            referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

            players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
            players_detections.class_id = team_classifier.predict(players_crops)
            goalkeepers_detections.class_id = resolve_goalkeepers_team_id(players_detections, goalkeepers_detections)
            referees_detections.class_id -= 1

            all_detections = sv.Detections.merge([players_detections, goalkeepers_detections, referees_detections])

            result_kp = keypoint_model.infer(frame, confidence=confidence)[0]
            key_points = sv.KeyPoints.from_inference(result_kp)

            filter_pts = key_points.confidence[0] > 0.5
            frame_reference_points = key_points.xy[0][filter_pts]
            pitch_reference_points = np.array(config.vertices)[filter_pts]

            transformer = ViewTransformer(source=frame_reference_points, target=pitch_reference_points)

            frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

            players_xy = all_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_players_xy = transformer.transform_points(points=players_xy)

            referees_xy = referees_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_referees_xy = transformer.transform_points(points=referees_xy)

            annotated_frame = draw_pitch(config)
            annotated_frame = draw_points_on_pitch(
                config=config, xy=pitch_ball_xy,
                face_color=sv.Color.WHITE, edge_color=sv.Color.BLACK,
                radius=10, pitch=annotated_frame
            )
            annotated_frame = draw_points_on_pitch(
                config=config, xy=pitch_players_xy[all_detections.class_id == 0],
                face_color=sv.Color.from_hex('00BFFF'), edge_color=sv.Color.BLACK,
                radius=16, pitch=annotated_frame
            )
            annotated_frame = draw_points_on_pitch(
                config=config, xy=pitch_players_xy[all_detections.class_id == 1],
                face_color=sv.Color.from_hex('FF1493'), edge_color=sv.Color.BLACK,
                radius=16, pitch=annotated_frame
            )
            annotated_frame = draw_points_on_pitch(
                config=config, xy=pitch_referees_xy,
                face_color=sv.Color.from_hex('FFD700'), edge_color=sv.Color.BLACK,
                radius=16, pitch=annotated_frame
            )

            video_sink.write_frame(annotated_frame)


def generate_voronoi_video(
    source_video_path,
    target_video_path,
    detection_model,
    keypoint_model,
    team_classifier,
    config,
    confidence=0.3
):
    """
    Génère une vidéo avec diagramme de Voronoï (zones de contrôle).
    """
    from sports.annotators.soccer import draw_pitch_voronoi_diagram
    
    BALL_ID = 0
    GOALKEEPER_ID = 1
    PLAYER_ID = 2
    REFEREE_ID = 3

    tracker = sv.ByteTrack()
    tracker.reset()

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    video_sink = sv.VideoSink(target_video_path, video_info=video_info)
    frame_generator = sv.get_video_frames_generator(source_video_path)

    with video_sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames, desc="Generating Voronoi video"):
            result = detection_model.predict(frame, conf=confidence)[0]
            detections = sv.Detections.from_ultralytics(result)

            ball_detections = detections[detections.class_id == BALL_ID]
            ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

            all_detections = detections[detections.class_id != BALL_ID]
            all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
            all_detections = tracker.update_with_detections(all_detections)

            goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
            players_detections = all_detections[all_detections.class_id == PLAYER_ID]

            players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
            players_detections.class_id = team_classifier.predict(players_crops)
            goalkeepers_detections.class_id = resolve_goalkeepers_team_id(players_detections, goalkeepers_detections)

            all_players = sv.Detections.merge([players_detections, goalkeepers_detections])

            result_kp = keypoint_model.infer(frame, confidence=confidence)[0]
            key_points = sv.KeyPoints.from_inference(result_kp)

            filter_pts = key_points.confidence[0] > 0.5
            frame_reference_points = key_points.xy[0][filter_pts]
            pitch_reference_points = np.array(config.vertices)[filter_pts]

            transformer = ViewTransformer(source=frame_reference_points, target=pitch_reference_points)

            frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

            players_xy = all_players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_players_xy = transformer.transform_points(points=players_xy)

            annotated_frame = draw_pitch(config, background_color=sv.Color.WHITE, line_color=sv.Color.BLACK)
            annotated_frame = draw_pitch_voronoi_diagram(
                config=config,
                team_1_xy=pitch_players_xy[all_players.class_id == 0],
                team_2_xy=pitch_players_xy[all_players.class_id == 1],
                team_1_color=sv.Color.from_hex('00BFFF'),
                team_2_color=sv.Color.from_hex('FF1493'),
                pitch=annotated_frame
            )

            annotated_frame = draw_points_on_pitch(
                config=config, xy=pitch_ball_xy,
                face_color=sv.Color.WHITE, edge_color=sv.Color.WHITE,
                radius=8, thickness=1, pitch=annotated_frame
            )
            annotated_frame = draw_points_on_pitch(
                config=config, xy=pitch_players_xy[all_players.class_id == 0],
                face_color=sv.Color.from_hex('00BFFF'), edge_color=sv.Color.WHITE,
                radius=16, thickness=1, pitch=annotated_frame
            )
            annotated_frame = draw_points_on_pitch(
                config=config, xy=pitch_players_xy[all_players.class_id == 1],
                face_color=sv.Color.from_hex('FF1493'), edge_color=sv.Color.WHITE,
                radius=16, thickness=1, pitch=annotated_frame
            )

            video_sink.write_frame(annotated_frame)