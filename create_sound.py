'''スケールおかしいけど機能はできた'''
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
import pygame
from scipy.spatial import KDTree

# RealSenseセットアップ
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # カラー画像のストリームを有効化
pipeline.start(config)

# Pygameの初期化
pygame.init()
pygame.mixer.init()

# 音声オブジェクトの作成
def create_sound():
    sound = pygame.mixer.Sound('touch.mp3')  # 音声ファイルを指定
    return sound

# 3Dデータの保存関数
def save_sandbox_shape():
    frames = pipeline.wait_for_frames()
    depth = frames.get_depth_frame()
    if not depth:
        raise ValueError("Depth frame is empty")

    # 深度データから点群を生成
    pc = rs.pointcloud()
    points = pc.calculate(depth)

    # Open3Dを使って点群を保存
    vertices = np.asanyarray(points.get_vertices())
    vertices = np.array([[v[0], v[1], v[2]] for v in vertices], dtype=np.float64)  # float64に変更
    pcd = o3d.geometry.PointCloud()
    try:
        pcd.points = o3d.utility.Vector3dVector(vertices)
    except Exception as e:
        print(f"Error while assigning points to PointCloud: {e}")
    
    o3d.io.write_point_cloud("sandbox.ply", pcd)
    return pcd

# 背景差分法の設定
fgbg = cv2.createBackgroundSubtractorMOG2()  # 背景差分法のインスタンス作成

# 深度データから手の位置を取得する関数
def get_hand_position(depth_frame, color_frame):
    # 深度画像をnumpy配列に変換
    depth_image = np.asanyarray(depth_frame.get_data())
    
    # カラーマップ化して深度データを可視化（グレースケールからカラーに変換）
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    
    # 背景差分法で手の輪郭を検出
    fgmask = fgbg.apply(depth_image)  # 背景差分法の適用
    fgmask = cv2.medianBlur(fgmask, 5)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hand_center_3d = None
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            # 手の輪郭を囲む矩形の描画
            x, y, w, h = cv2.boundingRect(contour)
            hand_center_2d = np.array([x + w / 2, y + h / 2])

            # 手の中心位置の深度情報（Z座標）を取得
            z = depth_image[int(hand_center_2d[1]), int(hand_center_2d[0])] / 1000.0  # ミリメートルをメートルに変換
            
            # 手の位置を3D座標に変換
            hand_center_3d = np.array([x + w / 2, y + h / 2, z])

            # 手の位置を2D画像上に描画
            cv2.rectangle(color_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(color_frame, (int(hand_center_2d[0]), int(hand_center_2d[1])), 5, (255, 0, 0), -1)

    return hand_center_3d, depth_colormap, color_frame

# 音の設定
sound = create_sound()
sound_playing = False

try:
    point_cloud = None
    points = None
    kdtree = None

    while True:
        # Pygameのイベント処理を追加
        pygame.event.pump()

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # フレームの有効性チェック
        if not depth_frame or not color_frame:
            print("Waiting for frames...")
            continue

        # カラーフレームをnumpy配列に変換（手の位置を描画するため）
        color_image = np.asanyarray(color_frame.get_data())

        # キー入力で3Dデータの保存
        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):  # 's'キーで砂場形状を保存
            point_cloud = save_sandbox_shape()
            points = np.asarray(point_cloud.points)
            kdtree = KDTree(points)
            print("3Dデータが保存されました")

        # 手の位置を深度データから取得し、可視化
        hand_position, depth_colormap, color_frame_with_hand = get_hand_position(depth_frame, color_image)

        # 深度画像とカラー画像を並べて表示
        images = np.hstack((color_frame_with_hand, depth_colormap))
        cv2.imshow('Depth and Hand Detection', images)

        # 3Dデータが保存されている場合に手の位置をチェック
        if hand_position is not None and kdtree is not None:
            distance, _ = kdtree.query(hand_position)
            print(f"手と3Dデータの距離: {distance} メートル")  # 距離を出力

            if distance < 0.5:
                if not sound_playing:
                    sound.play(-1)
                    sound_playing = True
            else:
                if sound_playing:
                    sound.stop()
                    sound_playing = False

        # 'q'キーで終了
        if key & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    pygame.mixer.quit()
    pygame.quit()
    cv2.destroyAllWindows()
