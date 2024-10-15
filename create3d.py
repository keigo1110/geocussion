'''3Dデータの保存ができた。背景差分では手の位置検出うまくいかず。'''
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
        print('save')
        pcd.points = o3d.utility.Vector3dVector(vertices)
        print('save1')
    except Exception as e:
        print(f"Error while assigning points to PointCloud: {e}")
    
    o3d.io.write_point_cloud("sandbox.ply", pcd)
    return pcd

# カメラキャプチャーと手の位置検出設定
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

# 音の設定
sound = create_sound()
sound_playing = False

try:
    point_cloud = None
    points = None
    kdtree = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # キー入力で3Dデータの保存
        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):  # 's'キーで砂場形状を保存
            point_cloud = save_sandbox_shape()
            points = np.asarray(point_cloud.points)
            kdtree = KDTree(points)
            print("3Dデータが保存されました")

        # 手の位置を検出
        fgmask = fgbg.apply(frame)
        fgmask = cv2.medianBlur(fgmask, 5)

        # 輪郭を検出
        contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                hand_center = np.array([x + w / 2, y + h / 2, 0])  # 仮のZ座標を設定

                # 3Dデータが保存されている場合に手の位置をチェック
                if kdtree is not None:
                    # KDTreeは3次元での比較が必要なため、hand_centerを3次元にする
                    distance, _ = kdtree.query(hand_center)
                    if distance < 1.0:  # しきい値を設定して音を鳴らすかどうかを決定
                        if not sound_playing:
                            sound.play(-1)
                            sound_playing = True
                    else:
                        if sound_playing:
                            sound.stop()
                            sound_playing = False

        # フレームを表示
        cv2.imshow('Hand Detection', frame)

        # 'q'キーで終了
        if key & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pipeline.stop()
    pygame.mixer.quit()
    pygame.quit()