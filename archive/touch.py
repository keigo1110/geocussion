'''機能は完全に満たした。高速化必須'''
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
import pygame
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time

# RealSenseセットアップ
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

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

# 背景差分法の設定
fgbg = cv2.createBackgroundSubtractorMOG2()

# 深度データから手の位置を取得する関数
def get_hand_position(depth_frame, color_image):
    # Convert depth frame to numpy array
    depth_image = np.asanyarray(depth_frame.get_data())
    
    # color_image is already a numpy array, so we don't need to convert it

    # Apply color thresholding to detect skin color
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply morphological operations to remove noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hand_center_3d = None
    if contours:
        # Find the largest contour (assumed to be the hand)
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 3000:  # Minimum area threshold
            # Find the center of the contour
            M = cv2.moments(max_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Get the depth at the center point
                depth = depth_frame.get_distance(cx, cy)

                # Convert 2D point to 3D point
                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                hand_center_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], depth)

                # Draw the contour and center point
                cv2.drawContours(color_image, [max_contour], 0, (0, 255, 0), 2)
                cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)

    # Create depth colormap
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    return hand_center_3d, depth_colormap, color_image

# 3D散布図とx,y,z面からの投影図の可視化関数
def visualize_3d_and_projections(vertices):
    # 点群からランダムにサンプリング（処理速度向上のため）
    sample_size = min(10000, vertices.shape[0])  # 最大10000点をサンプリング
    indices = np.random.choice(vertices.shape[0], sample_size, replace=False)
    sampled_vertices = vertices[indices]

    x = sampled_vertices[:, 0]
    y = sampled_vertices[:, 1]
    z = sampled_vertices[:, 2]

    # 図の作成
    fig = plt.figure(figsize=(20, 15))
    
    # 3D散布図
    ax_3d = fig.add_subplot(221, projection='3d')
    scatter = ax_3d.scatter(x, y, z, c=z, cmap='viridis', s=1)
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_title('3D Point Cloud')
    fig.colorbar(scatter, ax=ax_3d, label='Z value')

    # XY平面（上から見た図）
    ax_xy = fig.add_subplot(222)
    ax_xy.scatter(x, y, c=z, cmap='viridis', s=1)
    ax_xy.set_xlabel('X')
    ax_xy.set_ylabel('Y')
    ax_xy.set_title('XY Projection (Top View)')

    # XZ平面（前から見た図）
    ax_xz = fig.add_subplot(223)
    ax_xz.scatter(x, z, c=z, cmap='viridis', s=1)
    ax_xz.set_xlabel('X')
    ax_xz.set_ylabel('Z')
    ax_xz.set_title('XZ Projection (Front View)')

    # YZ平面（横から見た図）
    ax_yz = fig.add_subplot(224)
    ax_yz.scatter(y, z, c=z, cmap='viridis', s=1)
    ax_yz.set_xlabel('Y')
    ax_yz.set_ylabel('Z')
    ax_yz.set_title('YZ Projection (Side View)')

    plt.tight_layout()

    # imagesフォルダが存在しない場合は作成
    if not os.path.exists('images'):
        os.makedirs('images')

    # タイムスタンプを使用してユニークなファイル名を生成
    timestamp = int(time.time())
    filename = f'images/pointcloud_visualization_{timestamp}.png'
    plt.savefig(filename, dpi=300)
    plt.close()

# 音の設定
sound = create_sound()
sound_playing = False
last_distance = float('inf')  # 前回の距離を記録

last_visualization_time = time.time()

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

            if distance < 0.002 and not sound_playing:
                sound.play()
                sound_playing = True
            elif distance >= 0.002 and sound_playing:
                sound.stop()
                sound_playing = False

            # 距離の変化が大きい場合にのみ音を再生/停止
            if abs(distance - last_distance) > 0.0005:
                if distance < 0.002 and not sound_playing:
                    sound.play()
                    sound_playing = True
                elif distance >= 0.002 and sound_playing:
                    sound.stop()
                    sound_playing = False
                last_distance = distance

        # 3秒ごとに3D散布図とx,y,z面からの投影図を可視化
        current_time = time.time()
        if current_time - last_visualization_time >= 3.0:
            if point_cloud is not None:
                visualize_3d_and_projections(vertices)
                last_visualization_time = current_time

        if key & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    pygame.mixer.quit()
    pygame.quit()
    cv2.destroyAllWindows()