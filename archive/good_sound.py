import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
import pygame
from scipy.spatial import KDTree
import os
import time
import mediapipe as mp
from collections import deque
import tensorflow as tf
from pykalman import KalmanFilter

# RealSenseセットアップ
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Mediapipeの初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)  # 複数の手を検出するために2に設定

# 各手のカルマンフィルターの初期化
kalman_filters = [KalmanFilter(initial_state_mean=np.zeros(3), n_dim_obs=3) for _ in range(2)]
for kf in kalman_filters:
    kf.transition_matrices = np.eye(3)
    kf.observation_matrices = np.eye(3)
    kf.initial_state_covariance = np.eye(3)  # 初期状態共分散行列を設定

# Pygameの初期化
pygame.init()
pygame.mixer.init()

# 音声オブジェクトの作成
def create_sound():
    sound_close = pygame.mixer.Sound('music/touch.mp3')  # First sound
    sound_far = pygame.mixer.Sound('music/shukin.mp3')  # Second sound
    return sound_close, sound_far

# 3Dデータの保存関数
def save_sandbox_shape():
    frames = pipeline.wait_for_frames()
    depth = frames.get_depth_frame()
    if not depth:
        raise ValueError("Depth frame is empty")

    pc = rs.pointcloud()
    points = pc.calculate(depth)

    vertices = np.asanyarray(points.get_vertices())
    vertices = np.array([[v[0], v[1], v[2]] for v in vertices], dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    try:
        pcd.points = o3d.utility.Vector3dVector(vertices)
    except Exception as e:
        print(f"Error while assigning points to PointCloud: {e}")

    o3d.io.write_point_cloud("sandbox.ply", pcd)
    return pcd, vertices

# 音の設定
sound_close, sound_far = create_sound()
sound_playing = [None, None]  # 各手に対応する
last_distance = [float('inf'), float('inf')]  # 前回の距離を記録
last_play_time = [0, 0]  # 各手ごとの最後の再生時間を記録
cooldown_time = 1.0  # 1秒のクールダウンタイム

# Modify this part to check z value and play sound accordingly
def play_sound_based_on_z(hand_index, z_value):
    global sound_playing, last_play_time
    current_time = time.time()
    if current_time - last_play_time[hand_index] < cooldown_time:
        return  # クールダウンタイム中は音を再生しない

    if z_value < 0.5:  # If hand is close to the camera
        if sound_playing[hand_index] != 'close':
            sound_far.stop()
            sound_close.play()
            sound_playing[hand_index] = 'close'
            last_play_time[hand_index] = current_time
    else:  # If hand is far from the camera
        if sound_playing[hand_index] != 'far':
            sound_close.stop()
            sound_far.play()
            sound_playing[hand_index] = 'far'
            last_play_time[hand_index] = current_time

# 深度データから手の位置を取得する関数（Mediapipeを使用）
def get_hand_positions(depth_frame, color_image):
    hand_positions_3d = []

    # RGB画像をMediapipeに渡すために変換
    color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    results = hands.process(color_image_rgb)

    if results.multi_hand_landmarks:
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # 手のランドマークの中心（中指の根元）を取得
            cx, cy = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * color_image.shape[1]), \
                     int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * color_image.shape[0])

            # 座標が範囲外でないかを確認
            cx = max(0, min(cx, color_image.shape[1] - 1))
            cy = max(0, min(cy, color_image.shape[0] - 1))

            # 深度フレームから深度を取得
            depth = depth_frame.get_distance(cx, cy)

            # 2D座標を3D座標に変換
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            hand_position_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], depth)

            # カルマンフィルターを使用して平滑化
            smoothed_hand_position_3d = smooth_hand_positions_kalman(hand_index, hand_position_3d)
            hand_positions_3d.append(smoothed_hand_position_3d)

            # バウンディングボックスと中心点を描画
            cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)

    # 深度画像を取得
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    return hand_positions_3d, depth_colormap, color_image


def smooth_hand_positions_kalman(hand_index, current_position):
    if hand_index >= len(kalman_filters):
        # 手のインデックスがkalman_filtersの範囲外であれば、新しいカルマンフィルターを追加
        kf = KalmanFilter(initial_state_mean=np.zeros(3), n_dim_obs=3)
        kf.transition_matrices = np.eye(3)
        kf.observation_matrices = np.eye(3)
        kf.initial_state_covariance = np.eye(3)
        kalman_filters.append(kf)

    # フィルタリング実行
    filtered_state_mean, filtered_state_covariance = kalman_filters[hand_index].filter_update(
        kalman_filters[hand_index].initial_state_mean,
        kalman_filters[hand_index].initial_state_covariance,
        observation=current_position
    )
    kalman_filters[hand_index].initial_state_mean = filtered_state_mean  # 状態の更新
    kalman_filters[hand_index].initial_state_covariance = filtered_state_covariance  # 共分散行列の更新
    return filtered_state_mean


try:
    point_cloud = None
    vertices = None
    kdtree = None

    while True:
        pygame.event.pump()

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            print("Waiting for frames...")
            continue

        color_image = np.asanyarray(color_frame.get_data())

        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            point_cloud, vertices = save_sandbox_shape()
            points = np.asarray(point_cloud.points)
            kdtree = KDTree(points)
            print("3Dデータが保存されました")

        # 平滑化された手の位置を取得
        hand_positions, depth_colormap, color_frame_with_hand = get_hand_positions(depth_frame, color_image)

        images = np.hstack((color_frame_with_hand, depth_colormap))
        cv2.imshow('Depth and Hand Detection', images)

        if hand_positions and kdtree is not None:
            for hand_index, hand_position in enumerate(hand_positions):
                hand_z_value = hand_position[2]
                distance, _ = kdtree.query(hand_position)
                print(f"手{hand_index + 1}と3Dデータの距離: {distance:.6f} メートル")

                # 以下、既存の距離に基づく処理を続終
                if distance < 0.05:
                    play_sound_based_on_z(hand_index, hand_z_value)
                    sound_playing[hand_index] = True
                else:
                    # 音を停止する処理
                    sound_close.stop()
                    sound_far.stop()
                    sound_playing[hand_index] = False

        if key & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    pygame.mixer.quit()
    pygame.quit()
    cv2.destroyAllWindows()
