# パラメータ設定
DEPTH_WIDTH = 640  # 深度カメラの幅
DEPTH_HEIGHT = 480  # 深度カメラの高さ
FRAME_RATE = 30  # フレームレート
MAX_HANDS = 2  # 検出する手の最大数
COOLDOWN_TIME = 1.0  # 音の再生間隔（秒）
DISTANCE_THRESHOLD = 0.05  # 手と3Dデータの距離閾値（メートル）
CLOSE_HAND_THRESHOLD = 0.5  # 手が近いと判断する閾値（メートル）

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

# RealSenseカメラの初期設定
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, FRAME_RATE)
config.enable_stream(rs.stream.color, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.bgr8, FRAME_RATE)
pipeline.start(config)

# MediaPipeの手検出モジュールの初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=MAX_HANDS)  # 両手の検出を有効化

# 各手のカルマンフィルターの初期化（ノイズ除去用）
kalman_filters = [KalmanFilter(initial_state_mean=np.zeros(3), n_dim_obs=3) for _ in range(MAX_HANDS)]

for kf in kalman_filters:
    kf.transition_matrices = np.eye(3)  # 状態遷移行列の設定
    kf.observation_matrices = np.eye(3)  # 観測行列の設定
    kf.initial_state_covariance = np.eye(3)  # 初期状態の不確実性を表す共分散行列

# Pygameの初期化（音声再生用）
pygame.init()
pygame.mixer.init()

# 音声ファイルの読み込み関数
def create_sound():
    """音声ファイルを読み込み、サウンドオブジェクトを生成する"""
    sound_close = pygame.mixer.Sound('music/A00.mp3')  # 手が近い時の音
    sound_far = pygame.mixer.Sound('music/A04.mp3')   # 手が遠い時の音
    return sound_close, sound_far

# 3D点群データの保存関数
def save_sandbox_shape():
    """
    現在の深度フレームから3D点群を生成し、ノイズ除去処理を行った後PLYファイルとして保存する
    処理ステップ:
    1. 複数フレームの統合によるノイズ低減
    2. 統計的外れ値除去
    3. 法線推定と異常値除去
    4. DBSCAN クラスタリングによる孤立点除去
    5. ボクセルダウンサンプリングによる点群の均一化
    Returns:
        tuple: (処理済み点群, 頂点データ)
    """
    # パラメータ設定
    FRAMES_TO_AVERAGE = 5  # 平均化するフレーム数
    STATISTICAL_NB_NEIGHBORS = 20  # 統計的外れ値除去の近傍点数
    STATISTICAL_STD_RATIO = 2.0  # 標準偏差の倍率
    NORMAL_RADIUS = 0.03  # 法線推定の半径
    NORMAL_MAX_NN = 30  # 法線推定の最大近傍点数
    DBSCAN_EPS = 0.02  # DBSCANのイプシロンパラメータ
    DBSCAN_MIN_POINTS = 10  # DBSCANの最小点数
    VOXEL_SIZE = 0.005  # ボクセルダウンサンプリングのグリッドサイズ

    # 複数フレームの点群を統合
    accumulated_points = []
    for _ in range(FRAMES_TO_AVERAGE):
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        if not depth:
            continue
        # 点群の生成
        pc = rs.pointcloud()
        points = pc.calculate(depth)
        vertices = np.asanyarray(points.get_vertices())
        vertices = np.array([[v[0], v[1], v[2]] for v in vertices], dtype=np.float64)
        accumulated_points.append(vertices)
        time.sleep(0.1)  # フレーム間の短い待機時間
    # 点群の平均化
    averaged_vertices = np.mean(accumulated_points, axis=0)
    # Open3D点群オブジェクトの作成
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(averaged_vertices)
    # 統計的外れ値除去
    pcd, statistical_indices = pcd.remove_statistical_outlier(
        nb_neighbors=STATISTICAL_NB_NEIGHBORS,
        std_ratio=STATISTICAL_STD_RATIO
    )
    # 法線の推定
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS,
            max_nn=NORMAL_MAX_NN
        )
    )
    # 法線に基づく異常値除去（床面などの主要な面と大きく異なる法線を持つ点を除去）
    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)
    # 上向きの法線（床面）との角度が大きすぎる点を除去
    up_vector = np.array([0, 0, 1])
    normal_angles = np.abs(np.dot(normals, up_vector))
    valid_normal_mask = normal_angles > 0.3  # 閾値は調整可能
    pcd.points = o3d.utility.Vector3dVector(points[valid_normal_mask])
    # DBSCANクラスタリングで孤立点を除去
    labels = np.array(pcd.cluster_dbscan(
        eps=DBSCAN_EPS,
        min_points=DBSCAN_MIN_POINTS
    ))
    if len(labels) > 0:
        max_label = labels.max()
        print(f"DBSCAN クラスタリング: {max_label + 1} clusters found")
        # 最大のクラスターのみを保持
        if max_label >= 0:
            largest_cluster = labels == np.argmax(np.bincount(labels[labels >= 0]))
            pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[largest_cluster])
    # ボクセルダウンサンプリングで点群を均一化
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    try:
        # PLYファイルとして保存
        o3d.io.write_point_cloud("sandbox_processed.ply", pcd)
        print("処理済み点群を保存しました: sandbox_processed.ply")
        # 処理後の点群の統計情報を表示
        print(f"点の数: {len(np.asarray(pcd.points))}")
        return pcd, np.asarray(pcd.points)
    except Exception as e:
        print(f"点群データの保存中にエラーが発生しました: {e}")
        return None, None

# 音声再生の状態管理用の変数
sound_close, sound_far = create_sound()
sound_playing = [None, None]  # 各手の音声再生状態
last_distance = [float('inf'), float('inf')]  # 各手の前回の距離
last_play_time = [0, 0]  # 各手の最後の音声再生時刻

# 手の位置に基づいて音を再生する関数
def play_sound_based_on_z(hand_index, z_value):
    """
    手のZ座標（深さ）に基づいて適切な音を再生する
    Args:
        hand_index (int): 手のインデックス（0または1）
        z_value (float): 手のZ座標（深さ）
    """
    global sound_playing, last_play_time
    current_time = time.time()
    # クールダウン時間チェック
    if current_time - last_play_time[hand_index] < COOLDOWN_TIME:
        return
    # 手の位置に応じて音を再生
    if z_value < CLOSE_HAND_THRESHOLD:  # 手が近い場合
        if sound_playing[hand_index] != 'close':
            sound_far.stop()
            sound_close.play()
            sound_playing[hand_index] = 'close'
            last_play_time[hand_index] = current_time
    else:  # 手が遠い場合
        if sound_playing[hand_index] != 'far':
            sound_close.stop()
            sound_far.play()
            sound_playing[hand_index] = 'far'
            last_play_time[hand_index] = current_time

# 手の位置検出関数
def get_hand_positions(depth_frame, color_image):
    """
    カラー画像から手の位置を検出し、3D座標に変換する
    Args:
        depth_frame: RealSenseの深度フレーム
        color_image: カラー画像（numpy配列）
    Returns:
        tuple: (手の3D位置リスト, 深度カラーマップ, 手の位置を描画したカラー画像)
    """
    hand_positions_3d = []

    # MediaPipe用にRGB形式に変換
    color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    results = hands.process(color_image_rgb)
    if results.multi_hand_landmarks:
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # 中指の付け根の位置を手の中心として使用
            cx = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * color_image.shape[1])
            cy = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * color_image.shape[0])
            # 画像範囲内に収める
            cx = max(0, min(cx, color_image.shape[1] - 1))
            cy = max(0, min(cy, color_image.shape[0] - 1))
            # 深度値の取得
            depth = depth_frame.get_distance(cx, cy)
            # 2D座標から3D座標への変換
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            hand_position_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], depth)
            # カルマンフィルターで位置を平滑化
            smoothed_hand_position_3d = smooth_hand_positions_kalman(hand_index, hand_position_3d)
            hand_positions_3d.append(smoothed_hand_position_3d)
            # 検出した手の位置を画像上に表示
            cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)
    # 深度画像のカラーマップを生成
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    return hand_positions_3d, depth_colormap, color_image

# カルマンフィルターによる手の位置の平滑化
def smooth_hand_positions_kalman(hand_index, current_position):
    """
    カルマンフィルターを使用して手の位置データを平滑化する
    Args:
        hand_index (int): 手のインデックス
        current_position (numpy.array): 現在の手の位置
    Returns:
        numpy.array: 平滑化された手の位置
    """
    if hand_index >= len(kalman_filters):
        # 新しい手が検出された場合、カルマンフィルターを追加
        kf = KalmanFilter(initial_state_mean=np.zeros(3), n_dim_obs=3)
        kf.transition_matrices = np.eye(3)
        kf.observation_matrices = np.eye(3)
        kf.initial_state_covariance = np.eye(3)
        kalman_filters.append(kf)
    # カルマンフィルターによる状態更新
    filtered_state_mean, filtered_state_covariance = kalman_filters[hand_index].filter_update(
        kalman_filters[hand_index].initial_state_mean,
        kalman_filters[hand_index].initial_state_covariance,
        observation=current_position
    )
    # フィルターの状態を更新
    kalman_filters[hand_index].initial_state_mean = filtered_state_mean
    kalman_filters[hand_index].initial_state_covariance = filtered_state_covariance
    return filtered_state_mean

def calculate_hand_distances(hand_positions, kdtree, sound_close, sound_far, sound_playing, last_play_time):
    """
    手の位置と3Dデータとの距離を計算し、適切な音を再生する

    Args:
        hand_positions (list): 検出された手の3D位置のリスト
        kdtree (scipy.spatial.KDTree): 3D点群データのKD木
        sound_close (pygame.mixer.Sound): 近距離時の音声オブジェクト
        sound_far (pygame.mixer.Sound): 遠距離時の音声オブジェクト
        sound_playing (list): 各手の音声再生状態のリスト
        last_play_time (list): 各手の最後の音声再生時刻のリスト

    Returns:
        tuple: (更新された音声再生状態のリスト, 更新された最後の音声再生時刻のリスト)
    """
    if not hand_positions or kdtree is None:
        return sound_playing, last_play_time

    for hand_index, hand_position in enumerate(hand_positions):
        hand_z_value = hand_position[2]
        distance, _ = kdtree.query(hand_position)
        print(f"手{hand_index + 1}と3Dデータの距離: {distance:.6f} メートル")

        if distance < DISTANCE_THRESHOLD:
            play_sound_based_on_z(hand_index, hand_z_value)
            sound_playing[hand_index] = True
        else:
            # 距離が閾値を超えた場合は音を停止
            sound_close.stop()
            sound_far.stop()
            sound_playing[hand_index] = False

    return sound_playing, last_play_time

# メインループ
try:
    point_cloud = None
    vertices = None
    kdtree = None

    while True:
        pygame.event.pump()  # Pygameイベントの処理
        # フレームの取得
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            print("フレームの取得待機中...")
            continue

        # カラー画像の取得
        color_image = np.asanyarray(color_frame.get_data())

        # キー入力の確認
        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            # 's'キーで3D点群データを保存
            point_cloud, vertices = save_sandbox_shape()
            points = np.asarray(point_cloud.points)
            kdtree = KDTree(points)  # 近傍点検索用のKD木を構築
            print("3Dデータが保存されました")

        # 手の位置検出と表示
        hand_positions, depth_colormap, color_frame_with_hand = get_hand_positions(depth_frame, color_image)

        # 深度画像とカラー画像を横に並べて表示
        images = np.hstack((color_frame_with_hand, depth_colormap))
        cv2.imshow('Depth and Hand Detection', images)

        # 手の位置と3Dデータとの距離計算
        sound_playing, last_play_time = calculate_hand_distances(
            hand_positions,
            kdtree,
            sound_close,
            sound_far,
            sound_playing,
            last_play_time
        )

        # 'q'キーで終了
        if key & 0xFF == ord('q'):
            break
finally:
    # リソースの解放
    pipeline.stop()
    pygame.mixer.quit()
    pygame.quit()
    cv2.destroyAllWindows()