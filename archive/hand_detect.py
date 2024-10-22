'''手の検出精度アップ'''
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
import pygame
from scipy.spatial import KDTree
import os
import time
import mediapipe as mp

# RealSenseセットアップ
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Mediapipeの初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# Pygameの初期化
pygame.init()
pygame.mixer.init()

# 音声オブジェクトの作成
def create_sound():
    sound = pygame.mixer.Sound('touch.mp3')
    return sound

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
sound = create_sound()
sound_playing = False
last_distance = float('inf')  # 前回の距離を記録

# 深度データから手の位置を取得する関数（Mediapipeを使用）
def get_hand_position(depth_frame, color_image):
    hand_center_3d = None

    # RGB画像をMediapipeに渡すために変換
    color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    results = hands.process(color_image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 手のランドマークの中心（中指の根元）を取得
            cx, cy = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * color_image.shape[1]), \
                     int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * color_image.shape[0])

            # 座標が範囲外でないかを確認
            cx = max(0, min(cx, color_image.shape[1] - 1))  # x座標を範囲内に調整
            cy = max(0, min(cy, color_image.shape[0] - 1))  # y座標を範囲内に調整

            # 深度フレームから深度を取得
            depth = depth_frame.get_distance(cx, cy)

            # 2D座標を3D座標に変換
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            hand_center_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], depth)

            # バウンディングボックスと中心点を描画
            cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)
            break  # 最初の手だけを処理

    # 深度画像を取得
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    return hand_center_3d, depth_colormap, color_image


try:
    point_cloud = None
    points = None
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

        hand_position, depth_colormap, color_frame_with_hand = get_hand_position(depth_frame, color_image)

        images = np.hstack((color_frame_with_hand, depth_colormap))
        cv2.imshow('Depth and Hand Detection', images)

        if hand_position is not None and kdtree is not None:
            distance, _ = kdtree.query(hand_position)
            print(f"手と3Dデータの距離: {distance:.6f} メートル")

            if distance < 0.03 and not sound_playing:
                sound.play()
                sound_playing = True
            elif distance >= 0.03 and sound_playing:
                sound.stop()
                sound_playing = False

            # 距離の変化が大きい場合にのみ音を再生/停止
            if abs(distance - last_distance) > 0.01:
                if distance < 0.03 and not sound_playing:
                    sound.play()
                    sound_playing = True
                elif distance >= 0.03 and sound_playing:
                    sound.stop()
                    sound_playing = False
                last_distance = distance

        if key & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    pygame.mixer.quit()
    pygame.quit()
    cv2.destroyAllWindows()
