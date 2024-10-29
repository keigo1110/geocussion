# パラメータ設定
DEPTH_WIDTH = 640  # 深度カメラの幅
DEPTH_HEIGHT = 480  # 深度カメラの高さ
FRAME_RATE = 30  # フレームレート
MAX_HANDS = 2  # 検出する手の最大数
COOLDOWN_TIME = 1.0  # 音の再生間隔（秒）
DISTANCE_THRESHOLD = 0.05  # 手と3Dデータの距離閾値（メートル）
CLOSE_HAND_THRESHOLD = 0.5  # 手が近いと判断する閾値（メートル）
DETECT_RIGHT_HAND = True  # 右手を検出するかどうか
DETECT_LEFT_HAND = True   # 左手を検出するかどうか

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
hands = mp_hands.Hands(
    max_num_hands=MAX_HANDS,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

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
    sound_close = pygame.mixer.Sound('music/touch.mp3')  # 手が近い時の音
    sound_far = pygame.mixer.Sound('music/shukin.mp3')   # 手が遠い時の音
    return sound_close, sound_far

# 3D点群データの保存関数
def save_sandbox_shape():
    """
    現在の深度フレームから3D点群を生成し、ノイズ除去処理を行った後PLYファイルとして保存する
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

    # 法線に基づく異常値除去
    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)
    up_vector = np.array([0, 0, 1])
    normal_angles = np.abs(np.dot(normals, up_vector))
    valid_normal_mask = normal_angles > 0.3
    pcd.points = o3d.utility.Vector3dVector(points[valid_normal_mask])

    # DBSCANクラスタリング
    labels = np.array(pcd.cluster_dbscan(
        eps=DBSCAN_EPS,
        min_points=DBSCAN_MIN_POINTS
    ))

    if len(labels) > 0:
        max_label = labels.max()
        print(f"DBSCAN クラスタリング: {max_label + 1} clusters found")
        if max_label >= 0:
            largest_cluster = labels == np.argmax(np.bincount(labels[labels >= 0]))
            pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[largest_cluster])

    # ボクセルダウンサンプリング
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)

    try:
        # PLYファイルとして保存
        o3d.io.write_point_cloud("sandbox_processed.ply", pcd)
        print("処理済み点群を保存しました: sandbox_processed.ply")
        return pcd, np.asarray(pcd.points)
    except Exception as e:
        print(f"点群データの保存中にエラーが発生しました: {e}")
        return None, None

# 手の状態管理クラス
class HandState:
    def __init__(self):
        self.sound_playing = False
        self.last_play_time = 0
        self.last_distance = float('inf')

# 音声再生の状態管理用の変数
sound_close, sound_far = create_sound()
hand_states = {
    "Right": HandState(),
    "Left": HandState()
}

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

def get_hand_positions(depth_frame, color_image):
    """
    カラー画像から指定された手（右手/左手）の位置を検出し、3D座標に変換する
    Args:
        depth_frame: RealSenseの深度フレーム
        color_image: カラー画像（numpy配列）
    Returns:
        tuple: (手の3D位置リスト, 手の種類リスト, 深度カラーマップ, 手の位置を描画したカラー画像)
    """
    hand_positions_3d = []
    hand_types = []  # 検出された手の種類（右/左）を保存

    # MediaPipe用にRGB形式に変換
    color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    results = hands.process(color_image_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # 手の種類を判定（右手/左手）
            hand_type = handedness.classification[0].label

            # 設定に基づいて特定の手のみを処理
            if (hand_type == "Right" and not DETECT_RIGHT_HAND) or \
               (hand_type == "Left" and not DETECT_LEFT_HAND):
                continue

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
            hand_index = len(hand_positions_3d)
            smoothed_hand_position_3d = smooth_hand_positions_kalman(hand_index, hand_position_3d)

            hand_positions_3d.append(smoothed_hand_position_3d)
            hand_types.append(hand_type)

            # 検出した手の位置を画像上に表示（右手は赤、左手は青で表示）
            color = (0, 0, 255) if hand_type == "Right" else (255, 0, 0)
            cv2.circle(color_image, (cx, cy), 5, color, -1)
            cv2.putText(color_image, hand_type, (cx - 30, cy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 深度画像のカラーマップを生成
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    return hand_positions_3d, hand_types, depth_colormap, color_image

def play_sound_based_on_z(hand_type, z_value):
    """
    手のZ座標（深さ）に基づいて適切な音を再生する
    Args:
        hand_type (str): 手の種類（"Right" または "Left"）
        z_value (float): 手のZ座標（深さ）
    """
    state = hand_states[hand_type]
    current_time = time.time()

    # クールダウン時間チェック
    if current_time - state.last_play_time < COOLDOWN_TIME:
        return

    # 手の位置に応じて音を再生
    if z_value < CLOSE_HAND_THRESHOLD:  # 手が近い場合
        if not state.sound_playing:
            sound_far.stop()
            sound_close.play()
            state.sound_playing = True
            state.last_play_time = current_time
            print(f"{hand_type}の手が近づきました")
    else:  # 手が遠い場合
        if state.sound_playing:
            sound_close.stop()
            sound_far.play()
            state.sound_playing = False
            state.last_play_time = current_time
            print(f"{hand_type}の手が離れました")

def calculate_hand_distances(hand_positions, hand_types, kdtree):
    """
    手の位置と3Dデータとの距離を計算し、適切な音を再生する
    Args:
        hand_positions (list): 検出された手の3D位置のリスト
        hand_types (list): 検出された手の種類のリスト（"Right" または "Left"）
        kdtree (scipy.spatial.KDTree): 3D点群データのKD木
    """
    if not hand_positions or kdtree is None:
        return

    for hand_position, hand_type in zip(hand_positions, hand_types):
        if ((hand_type == "Right" and not DETECT_RIGHT_HAND) or
            (hand_type == "Left" and not DETECT_LEFT_HAND)):
            continue

        hand_z_value = hand_position[2]
        distance, _ = kdtree.query(hand_position)
        print(f"{hand_type}の手と3Dデータの距離: {distance:.6f} メートル")

        if distance < DISTANCE_THRESHOLD:
            play_sound_based_on_z(hand_type, hand_z_value)
        else:
            # 距離が閾値を超えた場合は音を停止
            if hand_states[hand_type].sound_playing:
                sound_close.stop()
                sound_far.stop()
                hand_states[hand_type].sound_playing = False

# メインループ
def main():
    """
    メインプログラムのループ処理
    キー操作:
    - 's': 3D点群データを保存
    - 'r': 右手の検出ON/OFF切り替え
    - 'l': 左手の検出ON/OFF切り替え
    - 'q': プログラム終了
    """
    global DETECT_RIGHT_HAND, DETECT_LEFT_HAND

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
                if point_cloud is not None:
                    points = np.asarray(point_cloud.points)
                    kdtree = KDTree(points)
                    print("3Dデータが保存されました")

            # 手の位置検出と表示
            hand_positions, hand_types, depth_colormap, color_frame_with_hand = get_hand_positions(depth_frame, color_image)

            # 手の位置と3Dデータとの距離計算
            calculate_hand_distances(hand_positions, hand_types, kdtree)

            # ウィンドウに検出状態を表示
            status_text = f"detecting: right hand{'ON' if DETECT_RIGHT_HAND else 'OFF'} | left hand{'ON' if DETECT_LEFT_HAND else 'OFF'}"
            cv2.putText(color_frame_with_hand, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 操作説明の表示
            help_text = [
                "key operation:",
                "s: save 3D data",
                "r: change detecting right hand",
                "l: change detecting left hand",
                "q: finish"
            ]
            for i, text in enumerate(help_text):
                cv2.putText(color_frame_with_hand, text, (10, 60 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 深度画像とカラー画像を横に並べて表示
            images = np.hstack((color_frame_with_hand, depth_colormap))
            cv2.imshow('Depth and Hand Detection', images)

            # キー入力による検出切り替え
            if key & 0xFF == ord('r'):  # 'r'キーで右手の検出を切り替え
                DETECT_RIGHT_HAND = not DETECT_RIGHT_HAND
                print(f"右手の検出: {'ON' if DETECT_RIGHT_HAND else 'OFF'}")
            elif key & 0xFF == ord('l'):  # 'l'キーで左手の検出を切り替え
                DETECT_LEFT_HAND = not DETECT_LEFT_HAND
                print(f"左手の検出: {'ON' if DETECT_LEFT_HAND else 'OFF'}")
            elif key & 0xFF == ord('q'):  # 'q'キーで終了
                break

    finally:
        # リソースの解放
        pipeline.stop()
        pygame.mixer.quit()
        pygame.quit()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()