'''高さによって音が変える仕組みを追加'''
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

# RealSenseセットアップ
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Mediapipeの初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# 過去5フレームの手の位置を保存するデック
hand_positions_history = deque(maxlen=5)

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
sound_playing = None
last_distance = float('inf')  # 前回の距離を記録

# Modify this part to check z value and play sound accordingly
def play_sound_based_on_z(z_value):
    global sound_playing
    if z_value < 0.5:  # If hand is close to the camera
        if sound_playing != 'close':
            sound_far.stop()
            sound_close.play()
            sound_playing = 'close'
    else:  # If hand is far from the camera
        if sound_playing != 'far':
            sound_close.stop()
            sound_far.play()
            sound_playing = 'far'

# 深度データから手の位置を取得する関数（Mediapipeを使用）
def get_hand_position(depth_frame, color_image):
    hand_position_3d = None

    # RGB画像をMediapipeに渡すために変換
    color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    results = hands.process(color_image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
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

            # 現在の手の位置をデックに追加し、平滑化
            smoothed_hand_position_3d = smooth_hand_positions(hand_position_3d)

            # バウンディングボックスと中心点を描画
            cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)
            break  # 最初の手だけを処理

    else:
        # 手が検出されなかった場合の処理
        smoothed_hand_position_3d = None

    # 深度画像を取得
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    return smoothed_hand_position_3d, depth_colormap, color_image


def smooth_hand_positions(current_position):
    hand_positions_history.append(current_position)
    if len(hand_positions_history) == hand_positions_history.maxlen:
        avg_position = np.mean(hand_positions_history, axis=0)
    else:
        avg_position = current_position  # データが十分でない場合は現在の位置を使用
    return avg_position


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
        hand_position, depth_colormap, color_frame_with_hand = get_hand_position(depth_frame, color_image)

        images = np.hstack((color_frame_with_hand, depth_colormap))
        cv2.imshow('Depth and Hand Detection', images)

        if hand_position is not None and kdtree is not None:
            distance, _ = kdtree.query(hand_position)
            hand_z_value = hand_position[2]
            print(f"手と3Dデータの距離: {distance:.6f} メートル")

            # 以下、既存の距離に基づく処理を継続
            if distance < 0.03:
                play_sound_based_on_z(hand_z_value)
                sound_playing = True
            else:
                # 音を停止する処理
                sound_close.stop()
                sound_far.stop()
                sound_playing = False

        if key & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    pygame.mixer.quit()
    pygame.quit()
    cv2.destroyAllWindows()