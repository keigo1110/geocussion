"""インタラクティブ砂場音響システム
手の検出と追跡コンポーネント
"""

import numpy as np
import mediapipe as mp
import cv2
import threading
import time
from collections import deque
from typing import List, Dict, Optional, Tuple
import logging
import pyrealsense2 as rs
from .core import SystemParameters, DEBUG_MODE

logger = logging.getLogger(__name__)
class MotionMetrics:
    """動作の計測と分析を行うクラス"""
    def __init__(self, history_size: int = 5):
        self.position_history = deque(maxlen=history_size)
        self.velocity_history = deque(maxlen=history_size)
        self.acceleration_history = deque(maxlen=history_size)
        self.last_position: Optional[np.ndarray] = None
        self.last_velocity: Optional[np.ndarray] = None
        self.last_time = time.time()
        
        # カルマンフィルター状態
        self._initialize_kalman_filter()

        # MediaPipeの初期化（同期的に行う）
        self._initialize_mediapipe()
        logger.info("HandTracker initialized")

    def _initialize_mediapipe(self):
        """MediaPipeの初期化（修正版）"""
        try:
            # MediaPipe Handsの設定
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,  # 検出感度を上げる
                min_tracking_confidence=0.5,
                model_complexity=1  # 0=Lite, 1=Full
            )
            
            # 描画ユーティリティの設定
            self.mp_draw = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            print("MediaPipe Hands initialized successfully")  # 直接プリント
            logger.info("MediaPipe Hands initialized successfully")
            
        except Exception as e:
            print(f"MediaPipe initialization error: {e}")  # 直接プリント
            logger.error(f"MediaPipe初期化エラー: {e}")
            raise

    def _initialize_kalman_filter(self):
        """カルマンフィルターの初期化"""
        self.state_initialized = False
        self.state_mean = np.zeros(3)
        self.state_covariance = np.eye(3)
        
        # カルマンフィルターパラメータ
        self.observation_matrix = np.eye(3)
        self.transition_matrix = np.eye(3)
        self.observation_covariance = 0.1 * np.eye(3)
        self.transition_covariance = 0.1 * np.eye(3)
    
    def _initialize_state(self, position: np.ndarray) -> bool:
        """カルマンフィルターの状態初期化"""
        try:
            if position is None or not np.all(np.isfinite(position)):
                return False
            
            self.state_mean = position.copy()
            self.state_covariance = np.eye(3)
            self.state_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"状態初期化エラー: {e}")
            return False

    def update(self, current_position: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """現在位置から速度と加速度を計算"""
        try:
            if current_position is None or not np.all(np.isfinite(current_position)):
                logger.debug("無効な位置データを検出")
                return None, None, None
            
            current_time = time.time()
            dt = current_time - self.last_time
            
            if dt <= 0:
                logger.debug("無効な時間差分を検出")
                return None, None, None
            
            # カルマンフィルタリング
            filtered_position = self._apply_kalman_filter(current_position)
            if filtered_position is None:
                return None, None, None
            
            # 速度と加速度の計算
            current_velocity = None
            current_acceleration = None
            
            if self.last_position is not None:
                current_velocity = self._calculate_velocity(filtered_position, dt)
                if current_velocity is not None and self.last_velocity is not None:
                    current_acceleration = self._calculate_acceleration(current_velocity, dt)
            
            # 履歴の更新
            self._update_histories(filtered_position, current_velocity, current_acceleration)
            
            # 状態の更新
            self.last_position = filtered_position
            self.last_velocity = current_velocity
            self.last_time = current_time
            
            if DEBUG_MODE:
                self._log_debug_info(filtered_position, current_velocity, current_acceleration)
            
            return filtered_position, current_velocity, current_acceleration
            
        except Exception as e:
            logger.error(f"モーション更新エラー: {e}")
            return None, None, None

    def _apply_kalman_filter(self, position: np.ndarray) -> Optional[np.ndarray]:
        """カルマンフィルターの適用"""
        try:
            if not self.state_initialized:
                if not self._initialize_state(position):
                    return None
            
            # 予測
            predicted_mean = np.dot(self.transition_matrix, self.state_mean)
            predicted_covariance = (
                np.dot(np.dot(self.transition_matrix, self.state_covariance),
                      self.transition_matrix.T) +
                self.transition_covariance
            )
            
            # 更新
            innovation_covariance = (
                np.dot(np.dot(self.observation_matrix, predicted_covariance),
                      self.observation_matrix.T) +
                self.observation_covariance
            )
            
            kalman_gain = np.dot(
                np.dot(predicted_covariance, self.observation_matrix.T),
                np.linalg.inv(innovation_covariance)
            )
            
            innovation = position - np.dot(self.observation_matrix, predicted_mean)
            filtered_position = predicted_mean + np.dot(kalman_gain, innovation)
            
            # 状態の更新
            self.state_mean = filtered_position
            self.state_covariance = predicted_covariance - np.dot(
                np.dot(kalman_gain, self.observation_matrix),
                predicted_covariance
            )
            
            return filtered_position[:3]  # 位置のみを返す
            
        except Exception as e:
            logger.error(f"カルマンフィルター適用エラー: {e}")
            return None

    def _calculate_velocity(self, position: np.ndarray, dt: float) -> Optional[np.ndarray]:
        """速度の計算"""
        try:
            velocity = (position - self.last_position) / dt
            return velocity if np.all(np.isfinite(velocity)) else None
        except Exception as e:
            logger.error(f"速度計算エラー: {e}")
            return None

    def _calculate_acceleration(self, velocity: np.ndarray, dt: float) -> Optional[np.ndarray]:
        """加速度の計算"""
        try:
            acceleration = (velocity - self.last_velocity) / dt
            return acceleration if np.all(np.isfinite(acceleration)) else None
        except Exception as e:
            logger.error(f"加速度計算エラー: {e}")
            return None

    def _update_histories(self, position: np.ndarray, velocity: Optional[np.ndarray], 
                         acceleration: Optional[np.ndarray]):
        """履歴の更新"""
        if position is not None:
            self.position_history.append(position)
        if velocity is not None:
            self.velocity_history.append(velocity)
        if acceleration is not None:
            self.acceleration_history.append(acceleration)

    def _log_debug_info(self, position: np.ndarray, velocity: Optional[np.ndarray], 
                       acceleration: Optional[np.ndarray]):
        """デバッグ情報のログ出力"""
        logger.debug(f"Position: {position}")
        if velocity is not None:
            logger.debug(f"Velocity: {np.linalg.norm(velocity)}")
        if acceleration is not None:
            logger.debug(f"Acceleration: {np.linalg.norm(acceleration)}")

    def get_average_metrics(self) -> Tuple[float, float, float]:
        """平均的な動作指標を取得"""
        try:
            # 速度の平均値計算
            avg_velocity = self._calculate_average_norm(self.velocity_history)
            
            # 加速度の平均値計算
            avg_acceleration = self._calculate_average_norm(self.acceleration_history)
            
            # 垂直方向への動きの割合を計算
            vertical_alignment = self._calculate_vertical_alignment()
            
            return avg_velocity, avg_acceleration, vertical_alignment
            
        except Exception as e:
            logger.error(f"メトリクス計算エラー: {e}")
            return 0.0, 0.0, 0.0

    def _calculate_average_norm(self, history: deque) -> float:
        """履歴データの平均ノルムを計算"""
        try:
            if not history:
                return 0.0
            valid_data = [v for v in history if np.all(np.isfinite(v))]
            if not valid_data:
                return 0.0
            norms = [np.linalg.norm(v) for v in valid_data]
            return float(np.mean(norms))
        except Exception as e:
            logger.error(f"平均ノルム計算エラー: {e}")
            return 0.0

    def _calculate_vertical_alignment(self) -> float:
        """垂直方向への動きの割合を計算"""
        try:
            if not self.velocity_history:
                return 0.0
            
            vertical_direction = np.array([0, 0, -1])
            alignments = []
            
            for velocity in self.velocity_history:
                if not np.all(np.isfinite(velocity)):
                    continue
                    
                v_norm = np.linalg.norm(velocity)
                if v_norm > 1e-6:  # ゼロ除算防止
                    alignment = np.dot(velocity, vertical_direction) / v_norm
                    if np.isfinite(alignment):
                        alignments.append(alignment)
            
            return float(np.mean(alignments)) if alignments else 0.0
            
        except Exception as e:
            logger.error(f"垂直アライメント計算エラー: {e}")
            return 0.0

    def reset(self):
        """状態のリセット"""
        try:
            self.position_history.clear()
            self.velocity_history.clear()
            self.acceleration_history.clear()
            self.last_position = None
            self.last_velocity = None
            self.last_time = time.time()
            self._initialize_kalman_filter()
        except Exception as e:
            logger.error(f"リセットエラー: {e}")

class HandTracker:
    """手の検出と追跡を行うクラス"""
    def __init__(self, params: SystemParameters):
        self.params = params
        self._lock = threading.Lock()
        self.processing_queue = deque(maxlen=5)
        self.last_process_time = time.time()
        self.process_interval = 1.0 / 30.0  # 30 FPS
        
        # フレームカウンタの初期化
        self.frame_count = 0
        
        # 状態変数の初期化
        self.last_valid_position = None
        self.initialization_complete = False
        
        # デバッグ情報の初期化
        self.debug_info = {
            'detections': 0,
            'successful_tracks': 0,
            'frame_times': deque(maxlen=30)
        }

        # MediaPipeの初期化
        self._initialize_mediapipe()
        logger.info("HandTracker initialized")

        
    def _initialize_mediapipe(self):
        """MediaPipeの初期化（修正版）"""
        try:
            # MediaPipe Handsの設定
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,  # 検出感度を上げる
                min_tracking_confidence=0.5,
                model_complexity=1  # 0=Lite, 1=Full
            )
            
            # 描画ユーティリティの設定
            self.mp_draw = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            print("MediaPipe Hands initialized successfully")
            logger.info("MediaPipe Hands initialized successfully")
            self.initialization_complete = True
            
        except Exception as e:
            print(f"MediaPipe initialization error: {e}")
            logger.error(f"MediaPipe初期化エラー: {e}")
            raise

    def _initialize_tracking_states(self):
        """追跡状態の初期化"""
        self.tracking_states = [{
            'id': None,
            'motion_metrics': MotionMetrics(),
            'last_detection_time': 0,
            'confidence': 0.0,
            'hand_type': None
        } for _ in range(self.params.MAX_HANDS)]
        
        self.next_track_id = 0

    def process_frame(self, color_image: np.ndarray, depth_frame: rs.frame,
                     depth_scale: float) -> Tuple[List[np.ndarray], List[str], np.ndarray]:
        """フレームの処理（修正版）"""
        try:
            # フレームカウンタの更新
            self.frame_count += 1
            start_time = time.time()
            
            # 入力チェック
            if color_image is None or depth_frame is None:
                logger.warning("Invalid input frames")
                return [], [], color_image.copy() if color_image is not None else np.zeros((480, 640, 3), dtype=np.uint8)

            # 初期化チェック
            if not hasattr(self, 'hands') or self.hands is None:
                logger.warning("HandTracker not fully initialized")
                return [], [], color_image.copy()

            # 画像の前処理
            if len(color_image.shape) == 2:
                color_image = cv2.cvtColor(color_image, cv2.COLOR_GRAY2BGR)
            color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            annotated_image = color_image.copy()

            # MediaPipeによる手の検出
            results = self.hands.process(color_image_rgb)

            # デバッグ情報
            if self.frame_count % 30 == 0:
                print(f"Processing frame {self.frame_count}")
                if results.multi_hand_landmarks:
                    print(f"Detected {len(results.multi_hand_landmarks)} hands")

            hand_positions_3d = []
            hand_types = []

            if results.multi_hand_landmarks:
                for idx, (hand_landmarks, handedness) in enumerate(
                    zip(results.multi_hand_landmarks, results.multi_handedness)):
                    
                    hand_type = handedness.classification[0].label
                    confidence = handedness.classification[0].score

                    # 検出設定に基づくフィルタリング
                    if hand_type == "Right" and not self.params.DETECT_RIGHT_HAND:
                        continue
                    if hand_type == "Left" and not self.params.DETECT_LEFT_HAND:
                        continue

                    try:
                        # ランドマークの描画
                        self._draw_landmarks(annotated_image, hand_landmarks, hand_type, confidence)
                    except Exception as draw_error:
                        print(f"Drawing error: {draw_error}")
                        logger.error(f"描画エラー: {draw_error}")

                    # 3D位置の取得
                    position = self._get_hand_position(
                        hand_landmarks, color_image.shape, depth_frame, depth_scale
                    )

                    if position is not None:
                        hand_positions_3d.append(position)
                        hand_types.append(hand_type)

                        # 位置情報を画像に描画
                        text_pos = (10, 30 + idx * 60)
                        cv2.putText(
                            annotated_image,
                            f"{hand_type}: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})",
                            text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )

            # パフォーマンス情報
            process_time = time.time() - start_time
            if self.frame_count % 30 == 0:
                print(f"Frame processing time: {process_time*1000:.1f}ms")

            return hand_positions_3d, hand_types, annotated_image

        except Exception as e:
            logger.error(f"フレーム処理エラー: {e}")
            print(f"Frame processing error: {e}")
            return [], [], color_image.copy()

    def _should_detect_hand(self, hand_type: str) -> bool:
        """手の種類に基づく検出判定"""
        return ((hand_type == "Right" and self.params.DETECT_RIGHT_HAND) or
                (hand_type == "Left" and self.params.DETECT_LEFT_HAND))

    def _get_hand_position(self, landmarks, image_shape, depth_frame, depth_scale
                          ) -> Optional[np.ndarray]:
        """手の3D位置の取得（修正版）"""
        try:
            height, width = image_shape[:2]
            
            # 人差し指の先端の位置を取得
            landmark = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # ピクセル座標の計算
            px = int(landmark.x * width)
            py = int(landmark.y * height)
            
            print(f"Pixel coordinates: ({px}, {py})")  # 直接プリント

            if not (0 <= px < width and 0 <= py < height):
                return None
            
            # 深度値の取得
            depth = depth_frame.get_distance(px, py)
            print(f"Depth value: {depth}m")  # 直接プリント

            if not (0.1 < depth <= 1.0):  # 10cm～1mの範囲
                return None
            
            # 3D座標の計算
            x = (px - width/2) * depth_scale * depth
            y = (py - height/2) * depth_scale * depth
            z = depth
            
            position = np.array([x, y, z], dtype=np.float32)
            
            # 値の検証
            if not np.all(np.isfinite(position)):
                return None
            
            return position
            
        except Exception as e:
            print(f"Hand position calculation error: {e}")  # 直接プリント
            logger.error(f"手の位置取得エラー: {e}")
            return None

    def _update_tracking_state(self, hand_type: str, position: np.ndarray, confidence: float) -> Optional[int]:
        """追跡状態の更新"""
        try:
            with self._lock:
                # 既存の追跡状態を検索
                state_idx = self._find_tracking_state(hand_type)
                
                if state_idx is None:
                    # 新しい追跡状態の作成
                    state_idx = self._create_new_tracking_state(hand_type)
                    if state_idx is None:
                        return None
                
                state = self.tracking_states[state_idx]
                
                # モーション解析の更新
                filtered_position, velocity, acceleration = state['motion_metrics'].update(position)
                
                if filtered_position is not None:
                    # 状態の更新
                    state.update({
                        'confidence': confidence,
                        'last_detection_time': time.time()
                    })
                    
                    if DEBUG_MODE:
                        self._log_tracking_update(state_idx, filtered_position, velocity)
                    
                return state_idx
                
        except Exception as e:
            logger.error(f"追跡状態更新エラー: {e}")
            return None

    def _find_tracking_state(self, hand_type: str) -> Optional[int]:
        """指定された手の種類に対応する追跡状態のインデックスを検索"""
        with self._lock:
            for i, state in enumerate(self.tracking_states):
                if (state['id'] is not None and 
                    state['hand_type'] == hand_type):
                    return i
        return None

    def _create_new_tracking_state(self, hand_type: str) -> Optional[int]:
        """新しい追跡状態の作成"""
        with self._lock:
            # 利用可能なスロットを探す
            for i, state in enumerate(self.tracking_states):
                if state['id'] is None:
                    state.update({
                        'id': self.next_track_id,
                        'hand_type': hand_type,
                        'motion_metrics': MotionMetrics(),
                        'last_detection_time': time.time(),
                        'confidence': 0.0
                    })
                    self.next_track_id += 1
                    return i
        return None

    def _check_timeouts(self):
        """追跡状態のタイムアウトチェック"""
        current_time = time.time()
        with self._lock:
            for state in self.tracking_states:
                if (state['id'] is not None and 
                    current_time - state['last_detection_time'] > 0.5):  # 0.5秒のタイムアウト
                    self._reset_tracking_state(state)

    def _reset_tracking_state(self, state: Dict):
        """追跡状態のリセット"""
        state.update({
            'id': None,
            'motion_metrics': MotionMetrics(),
            'last_detection_time': 0,
            'confidence': 0.0,
            'hand_type': None
        })

    def _draw_landmarks(self, image: np.ndarray, landmarks, hand_type: str, confidence: float):
            """ランドマークの描画（修正版）"""
            try:
                # 描画スタイルの定義
                landmark_style = self.mp_draw.DrawingSpec(
                    color=(255, 0, 0),  # 赤色
                    thickness=2,
                    circle_radius=4
                )
                connection_style = self.mp_draw.DrawingSpec(
                    color=(0, 255, 0),  # 緑色
                    thickness=2,
                    circle_radius=2
                )

                # 手のランドマークの描画
                self.mp_draw.draw_landmarks(
                    image,
                    landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_style,  # ランドマークのスタイル
                    connection_style  # 接続線のスタイル
                )
                
                # 手首のランドマークから情報表示位置を取得
                wrist = landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                pos = (
                    int(wrist.x * image.shape[1]),
                    int(wrist.y * image.shape[0])
                )

                # 手の種類と信頼度の表示
                text_color = (0, 0, 255) if hand_type == "Right" else (255, 0, 0)
                label_text = f"{hand_type} ({confidence:.2f})"

                # テキスト背景の描画（より見やすくするため）
                text_size = cv2.getTextSize(
                    label_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    2
                )[0]
                
                cv2.rectangle(
                    image,
                    (pos[0] - 30, pos[1] - 30),
                    (pos[0] - 30 + text_size[0], pos[1] - 10),
                    (0, 0, 0),
                    -1
                )

                # テキストの描画
                cv2.putText(
                    image,
                    label_text,
                    (pos[0] - 30, pos[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    text_color,
                    2
                )

                # 各指先の位置にIDを表示
                finger_tips = [
                    (self.mp_hands.HandLandmark.THUMB_TIP, "T"),
                    (self.mp_hands.HandLandmark.INDEX_FINGER_TIP, "I"),
                    (self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, "M"),
                    (self.mp_hands.HandLandmark.RING_FINGER_TIP, "R"),
                    (self.mp_hands.HandLandmark.PINKY_TIP, "P")
                ]

                for landmark_id, label in finger_tips:
                    landmark = landmarks.landmark[landmark_id]
                    pos = (
                        int(landmark.x * image.shape[1]),
                        int(landmark.y * image.shape[0])
                    )
                    cv2.putText(
                        image,
                        label,
                        pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        text_color,
                        1
                    )

                if DEBUG_MODE:
                    logger.debug(f"Landmarks drawn for {hand_type} hand")
                
            except Exception as e:
                logger.error(f"ランドマーク描画エラー: {e}")
                print(f"Landmark drawing error: {e}")  # 直接出力

    def _log_tracking_update(self, state_idx: int, position: np.ndarray, velocity: Optional[np.ndarray]):
        """追跡更新のデバッグログ"""
        try:
            state = self.tracking_states[state_idx]
            logger.debug(
                f"追跡更新 - ID: {state['id']}, "
                f"Type: {state['hand_type']}, "
                f"Pos: {position}, "
                f"Vel: {velocity if velocity is not None else 'N/A'}"
            )
        except Exception as e:
            logger.error(f"ログ出力エラー: {e}")

    def cleanup(self):
        """リソースの解放（修正版）"""
        try:
            with self._lock:
                if hasattr(self, 'hands') and self.hands:
                    self.hands.close()
                print("HandTracker resources cleaned up")  # 直接プリント
                logger.info("HandTrackerのリソースを解放しました")
                
        except Exception as e:
            print(f"Cleanup error: {e}")  # 直接プリント
            logger.error(f"HandTracker クリーンアップエラー: {e}")