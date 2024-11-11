"""インタラクティブ砂場音響システム
バージョン 2.0: 信頼性と安定性の向上
"""

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
from pykalman import KalmanFilter
import threading
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import logging
import sys
from contextlib import contextmanager
import warnings
import contextlib

# デバッグモードの設定
DEBUG_MODE = False

# プロトバッフの警告を抑制
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')

# ロギングの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sandbox_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class CameraConfig:
    """カメラの設定パラメータ"""
    width: int = 640
    height: int = 480
    fps: int = 30
    depth_format: rs.format = rs.format.z16
    color_format: rs.format = rs.format.bgr8
    depth_enabled: bool = True
    color_enabled: bool = True

@dataclass
class ProcessingConfig:
    """処理パラメータ"""
    frame_timeout: int = 5000  # ミリ秒
    retry_count: int = 5
    retry_delay: float = 0.1
    reset_interval: int = 300  # 秒
    warmup_frames: int = 30
    max_consecutive_failures: int = 10

class SystemParameters:
    """システムパラメータ管理クラス"""
    def __init__(self):
        # カメラ設定
        self.camera = CameraConfig()
        self.processing = ProcessingConfig()
        
        # 手の検出パラメータ
        self.MAX_HANDS = 2
        self.COOLDOWN_TIME = 1.0
        self.BASE_DISTANCE_THRESHOLD = 0.025
        self.BASE_CLOSE_HAND_THRESHOLD = 0.5
        
        # 動作検出パラメータ
        self.VELOCITY_THRESHOLD = 0.5
        self.ACCELERATION_THRESHOLD = 2.0
        self.DIRECTION_THRESHOLD = 0.7
        
        # 点群処理パラメータ
        self.FRAMES_TO_AVERAGE = 5
        self.STATISTICAL_NB_NEIGHBORS = 20
        self.STATISTICAL_STD_RATIO = 2.0
        self.NORMAL_RADIUS = 0.03
        self.NORMAL_MAX_NN = 30
        self.DBSCAN_EPS = 0.02
        self.DBSCAN_MIN_POINTS = 10
        self.VOXEL_SIZE = 0.005
        
        # 環境調整用パラメータ
        self.light_intensity_factor = 0.0
        self.sand_density_factor = 0.0
        
        # 検出フラグ
        self.DETECT_RIGHT_HAND = True
        self.DETECT_LEFT_HAND = True
    
    def get_adjusted_distance_threshold(self) -> float:
        """環境に応じて調整された距離閾値を取得"""
        return self.BASE_DISTANCE_THRESHOLD * (1 + self.light_intensity_factor)
    
    def get_adjusted_close_threshold(self) -> float:
        """環境に応じて調整された近接閾値を取得"""
        return self.BASE_CLOSE_HAND_THRESHOLD * (1 + self.sand_density_factor)

class ResourceManager:
    """リソース管理クラス"""
    def __init__(self):
        self._resources = []
        self._lock = threading.Lock()
    
    def register(self, resource: object, cleanup_method: str = 'cleanup'):
        """リソースの登録"""
        with self._lock:
            self._resources.append((resource, cleanup_method))
    
    def cleanup_all(self):
        """全リソースの解放"""
        with self._lock:
            for resource, cleanup_method in reversed(self._resources):
                try:
                    cleanup_func = getattr(resource, cleanup_method)
                    cleanup_func()
                except Exception as e:
                    logger.error(f"リソース解放エラー: {e}")
            self._resources.clear()

class MotionMetrics:
    """動作の計測と分析を行うクラス"""
    def __init__(self, history_size: int = 5):
        self.position_history = deque(maxlen=history_size)
        self.velocity_history = deque(maxlen=history_size)
        self.acceleration_history = deque(maxlen=history_size)
        self.last_position: Optional[np.ndarray] = None
        self.last_velocity: Optional[np.ndarray] = None
        self.last_time = time.time()
        
        # カルマンフィルターの状態
        self.state_initialized = False
        self.state_mean = np.zeros(3)
        self.state_covariance = np.eye(3)
        
        # カルマンフィルターパラメータ
        self.observation_matrix = np.eye(3)
        self.transition_matrix = np.eye(3)
        self.observation_covariance = 0.1 * np.eye(3)
        self.transition_covariance = 0.1 * np.eye(3)

    def _initialize_state(self, position: np.ndarray):
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
    
    def _kalman_predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """予測ステップ"""
        predicted_mean = np.dot(self.transition_matrix, self.state_mean)
        predicted_covariance = (
            np.dot(np.dot(self.transition_matrix, self.state_covariance),
                  self.transition_matrix.T) +
            self.transition_covariance
        )
        return predicted_mean, predicted_covariance
    
    def _kalman_update(self, measurement: np.ndarray, predicted_mean: np.ndarray,
                      predicted_covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """更新ステップ"""
        try:
            # カルマンゲインの計算
            innovation_covariance = (
                np.dot(np.dot(self.observation_matrix, predicted_covariance),
                      self.observation_matrix.T) +
                self.observation_covariance
            )
            
            kalman_gain = np.dot(
                np.dot(predicted_covariance, self.observation_matrix.T),
                np.linalg.inv(innovation_covariance)
            )
            
            # 状態の更新
            innovation = measurement - np.dot(self.observation_matrix, predicted_mean)
            updated_mean = predicted_mean + np.dot(kalman_gain, innovation)
            updated_covariance = predicted_covariance - np.dot(
                np.dot(kalman_gain, self.observation_matrix),
                predicted_covariance
            )
            
            return updated_mean, updated_covariance
            
        except Exception as e:
            logger.error(f"カルマン更新エラー: {e}")
            return predicted_mean, predicted_covariance

    def update(self, current_position: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        現在位置から速度と加速度を計算
        
        Args:
            current_position: 現在の3D位置
            
        Returns:
            Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]: 
            (フィルタリングされた位置, 速度, 加速度)
        """
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
            try:
                if not self.state_initialized:
                    if not self._initialize_state(current_position):
                        return None, None, None
                
                # 予測
                predicted_mean, predicted_covariance = self._kalman_predict()
                
                # 更新
                filtered_position, self.state_covariance = self._kalman_update(
                    current_position, predicted_mean, predicted_covariance
                )
                
                self.state_mean = filtered_position
                
                if not np.all(np.isfinite(filtered_position)):
                    logger.debug("フィルタリング後の位置が無効")
                    return None, None, None
                
            except Exception as e:
                logger.error(f"カルマンフィルタリングエラー: {e}")
                filtered_position = current_position
            
            # 位置履歴の更新
            self.position_history.append(filtered_position)
            current_velocity = None
            current_acceleration = None
            
            if self.last_position is not None:
                try:
                    # 速度計算
                    current_velocity = (filtered_position - self.last_position) / dt
                    if np.all(np.isfinite(current_velocity)):
                        self.velocity_history.append(current_velocity)
                        
                        if self.last_velocity is not None:
                            # 加速度計算
                            current_acceleration = (current_velocity - self.last_velocity) / dt
                            if np.all(np.isfinite(current_acceleration)):
                                self.acceleration_history.append(current_acceleration)
                    
                except Exception as e:
                    logger.error(f"速度/加速度計算エラー: {e}")
            
            # 状態の更新
            self.last_position = filtered_position
            self.last_velocity = current_velocity
            self.last_time = current_time
            
            if DEBUG_MODE:
                logger.debug(f"位置: {filtered_position}")
                if current_velocity is not None:
                    logger.debug(f"速度: {np.linalg.norm(current_velocity)}")
                if current_acceleration is not None:
                    logger.debug(f"加速度: {np.linalg.norm(current_acceleration)}")
            
            return filtered_position, current_velocity, current_acceleration
            
        except Exception as e:
            logger.error(f"モーション更新エラー: {e}")
            return None, None, None
    
    def get_average_metrics(self) -> Tuple[float, float, float]:
        """
        平均的な動作指標を取得
        
        Returns:
            Tuple[float, float, float]: 平均速度、平均加速度、垂直方向への動きの割合
        """
        try:
            avg_velocity = 0.0
            avg_acceleration = 0.0
            vertical_alignment = 0.0
            
            if self.velocity_history:
                try:
                    velocities = list(self.velocity_history)
                    if velocities:
                        # 速度の平均値計算
                        valid_velocities = [v for v in velocities if np.all(np.isfinite(v))]
                        if valid_velocities:
                            norms = [np.linalg.norm(v) for v in valid_velocities]
                            avg_velocity = float(np.mean(norms))
                            
                            # 垂直方向への動きを計算
                            vertical_direction = np.array([0, 0, -1])
                            alignments = []
                            for v in valid_velocities:
                                v_norm = np.linalg.norm(v)
                                if v_norm > 1e-6:  # ゼロ除算防止
                                    alignment = np.dot(v, vertical_direction) / v_norm
                                    if np.isfinite(alignment):
                                        alignments.append(alignment)
                            
                            if alignments:
                                vertical_alignment = float(np.mean(alignments))
                
                except Exception as e:
                    logger.error(f"速度メトリクス計算エラー: {e}")
            
            if self.acceleration_history:
                try:
                    accelerations = list(self.acceleration_history)
                    if accelerations:
                        valid_accelerations = [a for a in accelerations if np.all(np.isfinite(a))]
                        if valid_accelerations:
                            norms = [np.linalg.norm(a) for a in valid_accelerations]
                            avg_acceleration = float(np.mean(norms))
                            
                except Exception as e:
                    logger.error(f"加速度メトリクス計算エラー: {e}")
            
            return avg_velocity, avg_acceleration, vertical_alignment
            
        except Exception as e:
            logger.error(f"メトリクス計算エラー: {e}")
            return 0.0, 0.0, 0.0
    
    def reset(self):
        """状態のリセット"""
        try:
            self.position_history.clear()
            self.velocity_history.clear()
            self.acceleration_history.clear()
            self.last_position = None
            self.last_velocity = None
            self.last_time = time.time()
            self.state_initialized = False
            self.state_mean = np.zeros(3)
            self.state_covariance = np.eye(3)
        except Exception as e:
            logger.error(f"リセットエラー: {e}")

@contextmanager
def error_handler(operation_name: str):
    """エラーハンドリング用コンテキストマネージャ"""
    try:
        yield
    except Exception as e:
        logger.error(f"{operation_name}でエラーが発生: {e}", exc_info=True)
        raise

class PerformanceMonitor:
    """パフォーマンス監視クラス"""
    def __init__(self, window_size: int = 100):
        self.frame_times = deque(maxlen=window_size)
        self.last_time = time.perf_counter()
        self.fps = 0
        self._lock = threading.Lock()
    
    def update(self):
        """フレーム時間の更新"""
        current_time = time.perf_counter()
        with self._lock:
            self.frame_times.append(current_time - self.last_time)
            self.last_time = current_time
            
            if len(self.frame_times) >= 2:
                self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
    
    def get_fps(self) -> float:
        """現在のFPSを取得"""
        with self._lock:
            return self.fps
    
    def get_stats(self) -> Dict[str, float]:
        """パフォーマンス統計を取得"""
        with self._lock:
            if not self.frame_times:
                return {"fps": 0, "min_ms": 0, "max_ms": 0, "avg_ms": 0}
            
            frame_times_ms = [t * 1000 for t in self.frame_times]
            return {
                "fps": self.fps,
                "min_ms": min(frame_times_ms),
                "max_ms": max(frame_times_ms),
                "avg_ms": sum(frame_times_ms) / len(frame_times_ms)
            }

def check_system_requirements() -> Tuple[bool, str]:
    """
    システム要件のチェックを行い、結果と詳細メッセージを返す
    Returns:
        Tuple[bool, str]: (チェック結果, メッセージ)
    """
    try:
        messages = []
        
        # RealSenseデバイスの確認
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                return False, "RealSenseカメラが接続されていません"
            
            device = devices[0]
            messages.append(f"RealSenseカメラを検出: {device.get_info(rs.camera_info.name)}")
        except Exception as e:
            return False, f"RealSenseの初期化エラー: {e}"

        # 必要なディレクトリの存在確認と作成
        required_dirs = ['music']
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                try:
                    os.makedirs(dir_name)
                    messages.append(f"ディレクトリを作成: {dir_name}")
                except Exception as e:
                    return False, f"ディレクトリ作成エラー: {e}"

        # 必要な音声ファイルの確認とデフォルトファイルの作成
        sound_files = {
            'A00.mp3': b'', # ここにデフォルトの音声ファイルのバイナリデータを入れる
            'A04.mp3': b'', # ここにデフォルトの音声ファイルのバイナリデータを入れる
        }

        missing_files = []
        for filename, default_data in sound_files.items():
            filepath = os.path.join('music', filename)
            if not os.path.exists(filepath):
                missing_files.append(filename)

        if missing_files:
            return False, f"必要な音声ファイルが不足しています: {', '.join(missing_files)}\nmusic フォルダに以下のファイルを配置してください:\n- A00.mp3\n- A04.mp3"

        # PyGameの初期化チェック
        try:
            if not pygame.get_init():
                pygame.init()
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            messages.append("PyGameの初期化成功")
        except Exception as e:
            return False, f"PyGameの初期化エラー: {e}"

        # OpenCVのチェック
        try:
            messages.append(f"OpenCV バージョン: {cv2.__version__}")
        except Exception as e:
            return False, f"OpenCVの確認エラー: {e}"

        return True, "\n".join(messages)

    except Exception as e:
        return False, f"システム要件チェック中のエラー: {e}"

def initialize_system() -> Tuple[bool, Optional[str]]:
    """システムの初期化"""
    success, message = check_system_requirements()
    if not success:
        logger.error(f"システム要件チェック失敗: {message}")
        return False, message

    logger.info("システム要件チェック成功:")
    logger.info(message)
    return True, None

class SandboxSystem:
    """インタラクティブ砂場音響システムのメインクラス"""
    def __init__(self):
        try:
            self.params = SystemParameters()
            self.resource_manager = resource_manager
            self.performance_monitor = PerformanceMonitor()
            
            # システム要件のチェック
            success, message = check_system_requirements()
            if not success:
                print(f"\nシステム初期化エラー:\n{message}")
                print("\n必要な準備:")
                print("1. RealSenseカメラが正しく接続されていることを確認してください")
                print("2. music フォルダに必要な音声ファイルを配置してください")
                print("3. 必要なPythonパッケージがインストールされていることを確認してください")
                raise RuntimeError(message)
                
            self._initialize_system()
            
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"システム初期化エラー: {e}")

def initialize_system() -> Tuple[bool, Optional[str]]:
    """システムの初期化"""
    try:
        # システム要件のチェック
        if not check_system_requirements():
            return False, "システム要件を満たしていません"
        
        # PyGameの初期化
        try:
            pygame.init()
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        except pygame.error as e:
            return False, f"PyGameの初期化に失敗: {e}"
        
        # RealSenseのコンテキスト確認
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            return False, "RealSenseデバイスが接続されていません"
        
        logger.info("システムの初期化が完了しました")
        return True, None
        
    except Exception as e:
        return False, f"システムの初期化に失敗: {e}"

# グローバルなリソースマネージャーの作成
resource_manager = ResourceManager()

class RealSenseManager:
    """RealSenseカメラの管理クラス"""
    def __init__(self, params: SystemParameters):
        self.params = params
        self.pipeline: Optional[rs.pipeline] = None
        self.config: Optional[rs.config] = None
        self.align: Optional[rs.align] = None
        self.device: Optional[rs.device] = None
        self.depth_scale: float = 0.001  # デフォルト値
        self._lock = threading.Lock()
        self._frame_count = 0
        self._is_running = False
        self._last_reset_time = time.time()
    
    def initialize(self) -> bool:
        """カメラシステムの初期化"""
        try:
            with error_handler("カメラ初期化"):
                # パイプラインとコンフィグの作成
                self.pipeline = rs.pipeline()
                self.config = rs.config()
                
                # デバイスの検出と設定
                ctx = rs.context()
                devices = ctx.query_devices()
                if not devices:
                    raise RuntimeError("RealSenseデバイスが見つかりません")
                
                self.device = devices[0]
                logger.info(f"検出されたデバイス: {self.device.get_info(rs.camera_info.name)}")
                
                # デバイスのシリアル番号を取得
                device_serial = self.device.get_info(rs.camera_info.serial_number)
                self.config.enable_device(device_serial)
                
                # ストリームの設定
                self._configure_streams()
                
                # パイプラインの開始
                self.pipeline_profile = self.pipeline.start(self.config)
                
                # デプススケールの取得
                depth_sensor = self.pipeline_profile.get_device().first_depth_sensor()
                self.depth_scale = depth_sensor.get_depth_scale()
                
                # センサーの最適化
                self._optimize_sensors(depth_sensor)
                
                # アライメントオブジェクトの作成
                self.align = rs.align(rs.stream.color)
                
                # ウォームアップフレームの破棄
                self._warmup()
                
                self._is_running = True
                logger.info("カメラシステムの初期化が完了しました")
                return True
                
        except Exception as e:
            logger.error(f"カメラの初期化に失敗しました: {e}")
            self.cleanup()
            return False
    
    def _configure_streams(self):
        """ストリームの設定"""
        try:
            # 深度ストリーム
            self.config.enable_stream(
                rs.stream.depth,
                self.params.camera.width,
                self.params.camera.height,
                self.params.camera.depth_format,
                self.params.camera.fps
            )
            
            # カラーストリーム
            self.config.enable_stream(
                rs.stream.color,
                self.params.camera.width,
                self.params.camera.height,
                self.params.camera.color_format,
                self.params.camera.fps
            )
            
        except Exception as e:
            raise RuntimeError(f"ストリーム設定に失敗: {e}")
    
    def _optimize_sensors(self, depth_sensor: rs.sensor):
        """センサーの最適化"""
        try:
            # 深度センサーの設定
            if depth_sensor.supports(rs.option.visual_preset):
                depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy
            
            # レーザー出力の最適化
            if depth_sensor.supports(rs.option.laser_power):
                max_power = depth_sensor.get_option_range(rs.option.laser_power).max
                depth_sensor.set_option(rs.option.laser_power, max_power)
            
            # 深度信頼度しきい値の設定
            if depth_sensor.supports(rs.option.confidence_threshold):
                depth_sensor.set_option(rs.option.confidence_threshold, 3)
            
            # その他のセンサー固有の設定
            self._configure_advanced_settings(depth_sensor)
            
        except Exception as e:
            logger.warning(f"センサー最適化中の警告: {e}")
    
    def _configure_advanced_settings(self, depth_sensor: rs.sensor):
        """高度なセンサー設定"""
        try:
            # 利用可能なすべてのオプションを取得
            sensor_options = {
                option: depth_sensor.get_option_range(option)
                for option in [opt for opt in rs.option]
                if depth_sensor.supports(option)
            }
            
            # 特定のオプションの最適化
            option_settings = {
                rs.option.noise_filtering: 1,
                rs.option.post_processing_sharpening: 3,
                rs.option.pre_processing_sharpening: 5
            }
            
            for option, value in option_settings.items():
                if option in sensor_options:
                    opt_range = sensor_options[option]
                    clamped_value = min(max(value, opt_range.min), opt_range.max)
                    depth_sensor.set_option(option, clamped_value)
                    
        except Exception as e:
            logger.warning(f"高度なセンサー設定中の警告: {e}")
    
    def _warmup(self):
        """ウォームアップフレームの破棄"""
        try:
            for _ in range(self.params.processing.warmup_frames):
                frames = self.pipeline.wait_for_frames(
                    timeout_ms=self.params.processing.frame_timeout
                )
                if not frames:
                    raise RuntimeError("ウォームアップフレームの取得に失敗")
            logger.info("カメラウォームアップ完了")
            
        except Exception as e:
            logger.warning(f"ウォームアップ中の警告: {e}")
    
    def get_frames(self) -> Tuple[Optional[rs.frame], Optional[rs.frame]]:
        """フレームの取得"""
        if not self._is_running:
            return None, None
        
        with self._lock:
            try:
                # 定期的なリセットの確認
                current_time = time.time()
                if current_time - self._last_reset_time > self.params.processing.reset_interval:
                    logger.info("定期的なカメラリセットを実行")
                    self.reset()
                    self._last_reset_time = current_time
                
                # フレームの取得
                frames = self.pipeline.wait_for_frames(
                    timeout_ms=self.params.processing.frame_timeout
                )
                if not frames:
                    raise RuntimeError("フレームの取得に失敗")
                
                # フレームのアライメント
                aligned_frames = self.align.process(frames)
                
                # 深度フレームとカラーフレームの取得
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                # フレームの検証
                if not depth_frame or not color_frame:
                    raise RuntimeError("有効なフレームが取得できません")
                
                # フレームデータの検証
                if depth_frame.get_data_size() == 0 or color_frame.get_data_size() == 0:
                    raise RuntimeError("フレームデータが空です")
                
                self._frame_count += 1
                return depth_frame, color_frame
                
            except Exception as e:
                logger.error(f"フレーム取得エラー: {e}")
                return None, None
    
    def reset(self):
        """カメラシステムのリセット"""
        with self._lock:
            try:
                self._is_running = False
                if self.pipeline:
                    self.pipeline.stop()
                time.sleep(1)  # リセット待機
                self.initialize()
                
            except Exception as e:
                logger.error(f"カメラリセットエラー: {e}")
    
    def cleanup(self):
        """リソースの解放"""
        with self._lock:
            try:
                self._is_running = False
                if self.pipeline:
                    self.pipeline.stop()
                logger.info("カメラリソースを解放しました")
                
            except Exception as e:
                logger.error(f"カメラクリーンアップエラー: {e}")

class FrameProcessor:
    """フレーム処理クラス"""
    def __init__(self, params: SystemParameters):
        self.params = params
        self.depth_processor = rs.decimation_filter()
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()
        self.hole_filling_filter = rs.hole_filling_filter()
    
    def process_frames(self, depth_frame: rs.frame, color_frame: rs.frame) -> Tuple[np.ndarray, np.ndarray]:
        """フレームの処理"""
        try:
            # 深度フレームのフィルタリング
            filtered_depth = self._filter_depth_frame(depth_frame)
            
            # numpy配列への変換
            depth_image = np.asanyarray(filtered_depth.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            return depth_image, color_image
            
        except Exception as e:
            logger.error(f"フレーム処理エラー: {e}")
            return None, None
    
    def _filter_depth_frame(self, depth_frame: rs.frame) -> rs.frame:
        """深度フレームのフィルタリング"""
        try:
            # デシメーションフィルタ
            filtered = self.depth_processor.process(depth_frame)
            
            # スパシャルフィルタ
            filtered = self.spatial_filter.process(filtered)
            
            # テンポラルフィルタ
            filtered = self.temporal_filter.process(filtered)
            
            # ホールフィリング
            filtered = self.hole_filling_filter.process(filtered)
            
            return filtered
            
        except Exception as e:
            logger.warning(f"深度フィルタリング警告: {e}")
            return depth_frame
    
    def create_depth_colormap(self, depth_image: np.ndarray) -> np.ndarray:
        """深度画像のカラーマップ作成"""
        try:
            return cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
        except Exception as e:
            logger.error(f"深度カラーマップ作成エラー: {e}")
            return np.zeros_like(depth_image)

class HandTracker:
    """手の検出と追跡を行うクラス（非同期処理版）"""
    """手の検出と追跡を行うクラス（完全版）"""
    def __init__(self, params: SystemParameters):
        self.params = params
        self._initialize_mediapipe()
        self._initialize_kalman_filters()
        self._initialize_tracking_states()
        self._lock = threading.Lock()
        self.processing_queue = deque(maxlen=5)
        self.last_process_time = time.time()
        self.process_interval = 1.0 / 30.0  # 30 FPS
    
    def _initialize_mediapipe(self):
        """MediaPipe Handsの初期化"""
        try:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.params.MAX_HANDS,
                model_complexity=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
        except Exception as e:
            logger.error(f"MediaPipe初期化エラー: {e}")
            raise
    
    def _initialize_kalman_filters(self):
        """カルマンフィルターの初期化"""
        try:
            self.kalman_filters = []
            for _ in range(self.params.MAX_HANDS):
                kf = KalmanFilter(
                    initial_state_mean=np.zeros(9),  # 位置(3), 速度(3), 加速度(3)
                    n_dim_obs=3,                     # 観測は位置のみ
                    transition_matrices=self._create_transition_matrix(),
                    observation_matrices=self._create_observation_matrix(),
                    observation_covariance=0.1 * np.eye(3),
                    transition_covariance=0.1 * np.eye(9)
                )
                self.kalman_filters.append(kf)
        except Exception as e:
            logger.error(f"カルマンフィルター初期化エラー: {e}")
            raise
    
    def _initialize_tracking_states(self):
        """追跡状態の初期化"""
        self.tracking_states = [{
            'id': None,
            'position_history': deque(maxlen=10),
            'velocity_history': deque(maxlen=10),
            'acceleration_history': deque(maxlen=10),
            'last_detection_time': 0,
            'confidence': 0.0,
            'state_mean': np.zeros(9),
            'state_covar': np.eye(9),
            'hand_type': None
        } for _ in range(self.params.MAX_HANDS)]
        
        self.next_track_id = 0
    
    def _create_transition_matrix(self) -> np.ndarray:
        """状態遷移行列の作成"""
        dt = 1.0/30.0  # 想定フレームレート
        A = np.eye(9)
        # 位置の更新（速度による）
        A[0:3, 3:6] = np.eye(3) * dt
        # 速度の更新（加速度による）
        A[3:6, 6:9] = np.eye(3) * dt
        return A
    
    def _create_observation_matrix(self) -> np.ndarray:
        """観測行列の作成"""
        H = np.zeros((3, 9))
        H[0:3, 0:3] = np.eye(3)  # 位置のみ観測
        return H
    
    def _create_new_tracking_state(self, hand_data: Dict) -> int:
        """新しい追跡状態の作成"""
        with self._lock:
            for i, state in enumerate(self.tracking_states):
                if state['id'] is None:
                    state.update({
                        'id': self.next_track_id,
                        'hand_type': hand_data['type'],
                        'confidence': hand_data['confidence'],
                        'last_detection_time': time.time()
                    })
                    self.next_track_id += 1
                    return i
        return None

    def _predict_and_update(self, hand_idx: int, 
                           measurement: Optional[np.ndarray] = None) -> np.ndarray:
        """カルマンフィルターの予測と更新"""
        try:
            state = self.tracking_states[hand_idx]
            
            # 予測ステップ
            predicted_mean, predicted_covar = self.kalman_filters[hand_idx].filter_update(
                filtered_state_mean=state['state_mean'],
                filtered_state_covariance=state['state_covar'],
                observation=measurement
            )
            
            # 状態の更新
            state['state_mean'] = predicted_mean
            state['state_covar'] = predicted_covar
            
            # 位置、速度、加速度の抽出
            position = predicted_mean[0:3]
            velocity = predicted_mean[3:6]
            acceleration = predicted_mean[6:9]
            
            # 履歴の更新
            if measurement is not None:
                state['position_history'].append(position)
                state['velocity_history'].append(velocity)
                state['acceleration_history'].append(acceleration)
                state['last_detection_time'] = time.time()
            
            return position
            
        except Exception as e:
            logger.error(f"カルマンフィルター更新エラー: {e}")
            return measurement if measurement is not None else state['state_mean'][0:3]

    def _draw_landmarks(self, image: np.ndarray, landmarks, hand_type: str, confidence: float):
        """ランドマークの描画"""
        try:
            # ランドマークの描画
            self.mp_draw.draw_landmarks(
                image,
                landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # 手の種類と信頼度の表示
            landmark = landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            pos = (
                int(landmark.x * image.shape[1]),
                int(landmark.y * image.shape[0])
            )
            color = (0, 0, 255) if hand_type == "Right" else (255, 0, 0)
            cv2.putText(
                image,
                f"{hand_type} ({confidence:.2f})",
                (pos[0] - 30, pos[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        except Exception as e:
            logger.error(f"ランドマーク描画エラー: {e}")
    
    def process_frame(self, color_image: np.ndarray, depth_frame: rs.frame,
                     depth_scale: float) -> Tuple[List[np.ndarray], List[str], np.ndarray]:
        """フレームの処理（非同期版）"""
        try:
            current_time = time.time()
            if current_time - self.last_process_time < self.process_interval:
                return [], [], color_image.copy()
            
            self.last_process_time = current_time
            
            # 画像の前処理
            color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            annotated_image = color_image.copy()
            
            # MediaPipeによる手の検出
            results = self.hands.process(color_image_rgb)
            
            hand_positions_3d = []
            hand_types = []
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness):
                    
                    hand_type = handedness.classification[0].label
                    
                    # 設定に基づく手の種類のフィルタリング
                    if ((hand_type == "Right" and not self.params.DETECT_RIGHT_HAND) or
                        (hand_type == "Left" and not self.params.DETECT_LEFT_HAND)):
                        continue
                    
                    # 位置の取得
                    position = self._get_hand_position(
                        hand_landmarks, color_image.shape, depth_frame, depth_scale
                    )
                    
                    if position is not None:
                        if DEBUG_MODE:
                            logger.debug(f"{hand_type} 手の位置: {position}")
                            
                        hand_positions_3d.append(position)
                        hand_types.append(hand_type)
                        
                        # 描画
                        self._draw_landmarks(
                            annotated_image,
                            hand_landmarks,
                            hand_type,
                            handedness.classification[0].score
                        )
            
            if DEBUG_MODE and hand_positions_3d:
                logger.debug(f"検出された手の数: {len(hand_positions_3d)}")
            
            return hand_positions_3d, hand_types, annotated_image
            
        except Exception as e:
            logger.error(f"手の検出処理エラー: {e}")
            if DEBUG_MODE:
                import traceback
                logger.debug(traceback.format_exc())
            return [], [], color_image.copy()
    
    def _get_hand_position(self, landmarks, image_shape, depth_frame, depth_scale) -> Optional[np.ndarray]:
        """手の3D位置の取得"""
        try:
            height, width = image_shape[:2]
            
            # 人差し指の先端の位置を取得
            landmark = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # ピクセル座標の計算
            px = int(landmark.x * width)
            py = int(landmark.y * height)
            
            # 画像範囲内かチェック
            if not (0 <= px < width and 0 <= py < height):
                return None
            
            # 深度値の取得と検証
            try:
                depth = depth_frame.get_distance(px, py)
                if depth <= 0 or depth > 2.0:  # 2m以上は無効
                    return None
            except Exception as e:
                logger.error(f"深度値取得エラー: {e}")
                return None
            
            try:
                # カメラ座標系での3D位置
                x = (px - width/2) * depth_scale * depth
                y = (py - height/2) * depth_scale * depth
                z = depth
                
                position = np.array([x, y, z], dtype=np.float32)
                
                # 値の検証
                if not np.all(np.isfinite(position)):
                    return None
                
                return position
                
            except Exception as e:
                logger.error(f"座標変換エラー: {e}")
                return None
            
        except Exception as e:
            logger.error(f"手の位置取得エラー: {e}")
            return None

    def _reset_tracking_state(self, index: int):
        """追跡状態のリセット"""
        with self._lock:
            if 0 <= index < len(self.tracking_states):
                self.tracking_states[index] = {
                    'id': None,
                    'position_history': deque(maxlen=10),
                    'velocity_history': deque(maxlen=10),
                    'acceleration_history': deque(maxlen=10),
                    'last_detection_time': 0,
                    'confidence': 0.0,
                    'state_mean': np.zeros(9),
                    'state_covar': np.eye(9),
                    'hand_type': None
                }

    def _update_tracking_states(self, detected_hands: List[Dict]):
        """追跡状態の更新（修正版）"""
        try:
            with self._lock:
                # 各追跡状態の更新フラグを初期化
                updated_states = [False] * self.params.MAX_HANDS
                
                # 検出された手の処理
                for hand_data in detected_hands:
                    state_idx = self._find_tracking_state(hand_data['type'])
                    if state_idx is not None:
                        # 既存の追跡状態の更新
                        state = self.tracking_states[state_idx]
                        state['confidence'] = hand_data['confidence']
                        state['last_detection_time'] = time.time()
                        updated_states[state_idx] = True
                    else:
                        # 新しい追跡状態の作成
                        new_idx = self._create_new_tracking_state(hand_data)
                        if new_idx is not None:
                            updated_states[new_idx] = True
                
                # 更新されなかった状態の処理
                current_time = time.time()
                for i, updated in enumerate(updated_states):
                    if not updated:
                        state = self.tracking_states[i]
                        if (state['id'] is not None and 
                            current_time - state['last_detection_time'] > 1.0):
                            self._reset_tracking_state(i)
                            
        except Exception as e:
            logger.error(f"追跡状態更新エラー: {e}")
    
    def _find_tracking_state(self, hand_type: str) -> Optional[int]:
        """指定された手の種類に対応する追跡状態のインデックスを検索"""
        with self._lock:
            for i, state in enumerate(self.tracking_states):
                if (state['id'] is not None and 
                    state.get('hand_type') == hand_type):
                    return i
        return None
    
    def _predict_missing_hands(self):
        """未検出の手の状態予測"""
        current_time = time.time()
        for state in self.tracking_states:
            if (state['id'] is not None and 
                current_time - state['last_detection_time'] < 0.5):  # 0.5秒以内の消失のみ予測
                self._predict_and_update(
                    self._find_tracking_state(state.get('hand_type'))
                )
    
    def _draw_detection_results(self, image: np.ndarray, landmarks, 
                              hand_type: str, confidence: float):
        """検出結果の描画"""
        try:
            # ランドマークの描画
            self.mp_draw.draw_landmarks(
                image,
                landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # 手の種類とconfidenceの表示
            landmark = landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            pos = (
                int(landmark.x * image.shape[1]),
                int(landmark.y * image.shape[0])
            )
            color = (0, 0, 255) if hand_type == "Right" else (255, 0, 0)
            cv2.putText(
                image,
                f"{hand_type} ({confidence:.2f})",
                (pos[0] - 30, pos[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
            
        except Exception as e:
            logger.error(f"検出結果描画エラー: {e}")
    
    def cleanup(self):
        """リソースの解放"""
        try:
            self.hands.close()
        except Exception as e:
            logger.error(f"HandTracker クリーンアップエラー: {e}")

class AudioManager:
    """音声システム全体の管理クラス"""
    def __init__(self):
        self._lock = threading.Lock()
        self._initialized = False
        self._channels: Dict[int, pygame.mixer.Channel] = {}
        self._sounds: Dict[str, pygame.mixer.Sound] = {}
        
    def initialize(self) -> bool:
        """音声システムの初期化"""
        try:
            with self._lock:
                if not pygame.mixer.get_init():
                    pygame.mixer.init(
                        frequency=44100,
                        size=-16,
                        channels=2,
                        buffer=512
                    )
                
                # チャンネルの割り当て
                pygame.mixer.set_num_channels(16)  # 十分な数のチャンネルを確保
                
                self._initialized = True
                logger.info("音声システムを初期化しました")
                return True
                
        except Exception as e:
            logger.error(f"音声システムの初期化に失敗: {e}")
            return False
    
    def load_sound(self, sound_id: str, file_path: str) -> bool:
        """音声ファイルの読み込み"""
        try:
            with self._lock:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"音声ファイルが見つかりません: {file_path}")
                
                self._sounds[sound_id] = pygame.mixer.Sound(file_path)
                return True
                
        except Exception as e:
            logger.error(f"音声ファイルの読み込みに失敗 ({sound_id}): {e}")
            return False
    
    def get_channel(self, channel_id: int) -> Optional[pygame.mixer.Channel]:
        """専用チャンネルの取得"""
        try:
            with self._lock:
                if channel_id not in self._channels:
                    channel = pygame.mixer.Channel(channel_id)
                    self._channels[channel_id] = channel
                return self._channels[channel_id]
                
        except Exception as e:
            logger.error(f"チャンネル取得エラー: {e}")
            return None
    
    def cleanup(self):
        """リソースの解放"""
        try:
            with self._lock:
                for channel in self._channels.values():
                    channel.stop()
                self._channels.clear()
                self._sounds.clear()
                pygame.mixer.quit()
                self._initialized = False
                
        except Exception as e:
            logger.error(f"音声システムのクリーンアップに失敗: {e}")

class SoundController:
    """個別の音声制御クラス"""
    def __init__(self, audio_manager: AudioManager, params: SystemParameters,
                 hand_type: str, channel_id: int):
        self.audio_manager = audio_manager
        self.params = params
        self.hand_type = hand_type
        self.channel_id = channel_id
        self.motion_metrics = MotionMetrics()
        self._lock = threading.Lock()
        self.update_interval = 1.0 / 60.0  # 60 Hz更新
        self.last_update_time = time.time()
        self._initialize_sound_state()
        
        # 音声パラメータ
    def _initialize_sound_state(self):
        """音声状態の初期化"""
        self.current_volume = 0.0
        self.target_volume = 0.0
        self.fade_duration = 0.1
        self.is_playing = False
        self.is_in_contact = False
        self.last_contact_time = 0
        self.last_detection_time = time.time()
        self.fade_duration = 0.1
        
        # 状態管理
        self.is_playing = False
        self.is_in_contact = False
        self.last_contact_time = 0
        self.contact_cooldown = 0.5
        self.last_detection_time = time.time()
        self.no_detection_timeout = 0.5
        
        # 音声効果用のパラメータ
        self.resonance_factor = 1.0
        self.pitch_shift = 1.0
        self.effect_params = {
            'low_pass_cutoff': 1000,
            'high_pass_cutoff': 20,
            'reverb_amount': 0.3
        }
        
        # モーション分析用
        self.motion_metrics = MotionMetrics()
        
        self._initialize_sounds()
    
    def _initialize_sounds(self):
        """音声の初期化"""
        sound_files = {
            'close': f'music/A05.mp3',
            'far': f'music/A06.mp3',
            'impact': f'music/A07.mp3'
        }

        for sound_id, file_path in sound_files.items():
            if not self.audio_manager.load_sound(f"{self.hand_type}_{sound_id}", file_path):
                logger.warning(f"{sound_id} 音声の読み込みに失敗しました")
    
    def _detect_hit(self, distance: float, motion_metrics: Tuple[float, float, float]) -> Tuple[bool, Optional[float]]:
        """
        叩き動作の検出と強度計算（修正版）
        """
        try:
            if distance is None:
                return False, None
                
            avg_velocity, avg_acceleration, vertical_alignment = motion_metrics
            
            # 基本的な距離チェック
            if distance > self.params.get_adjusted_distance_threshold():
                return False, None
            
            # 各指標のチェック
            velocity_check = avg_velocity >= self.params.VELOCITY_THRESHOLD
            acceleration_check = avg_acceleration >= self.params.ACCELERATION_THRESHOLD
            direction_check = vertical_alignment >= self.params.DIRECTION_THRESHOLD
            
            # すべての条件を満たす場合のみヒットとして検出
            is_hit = velocity_check and acceleration_check and direction_check
            
            if is_hit:
                # ヒット強度の計算
                velocity_factor = min(1.0, max(0.0, avg_velocity / (self.params.VELOCITY_THRESHOLD * 2)))
                acceleration_factor = min(1.0, max(0.0, avg_acceleration / (self.params.ACCELERATION_THRESHOLD * 2)))
                intensity = (velocity_factor + acceleration_factor) / 2
                
                if not np.isfinite(intensity):
                    return False, None
                    
                return True, intensity
            
            return False, None
            
        except Exception as e:
            logger.error(f"ヒット検出エラー: {e}")
            return False, None

    def _calculate_volume(self, distance: float, metrics: Tuple[float, float, float]) -> float:
        """音量の計算"""
        try:
            if distance > self.params.get_adjusted_distance_threshold() * 2:
                return 0.0
            
            avg_velocity, avg_acceleration, _ = metrics
            
            # 基本音量（距離に基づく）
            base_volume = 1.0 - (distance / (self.params.get_adjusted_distance_threshold() * 2))
            base_volume = max(0.0, min(0.8, base_volume))
            
            # 動きに基づく追加音量
            motion_factor = min(1.0, (avg_velocity / self.params.VELOCITY_THRESHOLD +
                                    avg_acceleration / self.params.ACCELERATION_THRESHOLD) / 4)
            
            return min(1.0, base_volume + motion_factor * 0.2)
            
        except Exception as e:
            logger.error(f"音量計算エラー: {e}")
            return 0.0
   
    def _handle_no_detection(self):
        """検出なし時の処理"""
        try:
            current_time = time.time()
            if current_time - self.last_detection_time > self.no_detection_timeout:
                self.target_volume = 0.0
                if self.is_playing:
                    channel = self.audio_manager.get_channel(self.channel_id)
                    if channel:
                        channel.stop()
                    self.is_playing = False
                self.motion_metrics = MotionMetrics()
                self.is_in_contact = False
        except Exception as e:
            logger.error(f"検出なし処理エラー: {e}")

    def update_sound(self, distance: Optional[float], current_position: Optional[np.ndarray]):
        """サウンド状態の更新（堅牢性強化版）"""
        try:
            current_time = time.time()
            if current_time - self.last_update_time < self.update_interval:
                return
            
            with self._lock:
                self.last_update_time = current_time
                
                # 無効な入力値のチェック
                if distance is None or current_position is None:
                    self._handle_no_detection()
                    return
                
                # 距離の有効性チェック
                if not np.isfinite(distance) or distance < 0:
                    self._handle_no_detection()
                    return
                
                # 位置の有効性チェック
                if not np.all(np.isfinite(current_position)):
                    self._handle_no_detection()
                    return
                
                # 有効な検出として処理
                self.last_detection_time = current_time
                
                try:
                    # モーション解析
                    filtered_position, velocity, acceleration = self.motion_metrics.update(current_position)
                    if filtered_position is None:
                        self._handle_no_detection()
                        return
                    
                    motion_metrics = self.motion_metrics.get_average_metrics()
                    
                    # ヒット検出と音量計算
                    is_hit, intensity = self._detect_hit(distance, motion_metrics)
                    
                    if is_hit and intensity is not None:
                        time_since_last_contact = current_time - self.last_contact_time
                        if time_since_last_contact > self.contact_cooldown:
                            self.target_volume = self._adjust_volume_for_hit(intensity)
                            self.last_contact_time = current_time
                            if DEBUG_MODE:
                                logger.debug(
                                    f"{self.hand_type} Hit! "
                                    f"Intensity: {intensity:.2f}, "
                                    f"Volume: {self.target_volume:.2f}"
                                )
                    else:
                        self.target_volume = self._calculate_ambient_volume(distance)
                    
                    # 音量の更新
                    self._update_volume()
                    
                except Exception as e:
                    logger.error(f"モーション処理エラー: {e}")
                    self._handle_no_detection()
                
        except Exception as e:
            logger.error(f"サウンド更新エラー: {e}")
            self._handle_no_detection()
    
    def _calculate_sound_effects(self, distance: float, motion_metrics: Tuple[float, float, float]):
        """音響効果パラメータの計算"""
        try:
            avg_velocity, avg_acceleration, _ = motion_metrics
            
            # 共鳴係数の計算（距離に基づく）
            self.resonance_factor = 1.0 - (distance / self.params.BASE_DISTANCE_THRESHOLD)
            self.resonance_factor = max(0.0, min(1.0, self.resonance_factor))
            
            # ピッチシフトの計算（速度に基づく）
            velocity_factor = avg_velocity / self.params.VELOCITY_THRESHOLD
            self.pitch_shift = 1.0 + (velocity_factor * 0.2)  # 最大20%のピッチ変化
            
            # エフェクトパラメータの更新
            self.effect_params.update({
                'low_pass_cutoff': 1000 + (avg_acceleration * 500),  # 加速度に基づくフィルター
                'reverb_amount': 0.3 + (self.resonance_factor * 0.2)  # 共鳴に基づくリバーブ
            })
            
        except Exception as e:
            logger.error(f"エフェクト計算エラー: {e}")
    
    def _adjust_volume_for_hit(self, intensity: float) -> float:
        """ヒット時の音量調整"""
        try:
            if intensity is None:
                return 0.0
                
            # 基本音量
            base_volume = intensity * 0.8  # 最大音量の80%まで
            
            # 共鳴による増幅
            resonance_boost = self.resonance_factor * 0.2  # 最大20%の増幅
            
            # 最終音量の計算と制限
            final_volume = min(1.0, base_volume + resonance_boost)
            
            return final_volume
            
        except Exception as e:
            logger.error(f"音量調整エラー: {e}")
            return 0.0
    
    def _calculate_ambient_volume(self, distance: float) -> float:
        """環境音の音量計算"""
        try:
            if not np.isfinite(distance):
                return 0.0
            
            if distance > self.params.BASE_DISTANCE_THRESHOLD * 2:
                return 0.0
            
            # 距離に基づく減衰
            attenuation = 1.0 - (distance / (self.params.BASE_DISTANCE_THRESHOLD * 2))
            return float(max(0.0, min(0.3, attenuation * 0.3)))
            
        except Exception as e:
            logger.error(f"環境音量計算エラー: {e}")
            return 0.0
    
    def _play_impact_sound(self, intensity: float):
        """衝撃音の再生"""
        try:
            channel = self.audio_manager.get_channel(self.channel_id)
            if channel:
                sound = self.audio_manager._sounds.get(f"{self.hand_type}_impact")
                if sound:
                    channel.set_volume(intensity)
                    channel.play(sound)
                    
        except Exception as e:
            logger.error(f"衝撃音再生エラー: {e}")
    
    def _update_volume(self):
        """音量の更新"""
        try:
            current_time = time.time()
            delta_time = current_time - self.last_update_time
            
            if delta_time <= 0:
                return
            
            # クロスフェード
            volume_change = (self.target_volume - self.current_volume)
            fade_factor = min(delta_time / self.fade_duration, 1.0)
            self.current_volume += volume_change * fade_factor
            self.current_volume = max(0.0, min(1.0, self.current_volume))
            
            # 音の再生制御
            channel = self.audio_manager.get_channel(self.channel_id)
            if channel:
                if self.current_volume > 0.001:
                    if not self.is_playing:
                        sound_id = f"{self.hand_type}_close"
                        sound = self.audio_manager._sounds.get(sound_id)
                        if sound:
                            channel.play(sound)
                            self.is_playing = True
                    channel.set_volume(float(self.current_volume))
                else:
                    if self.is_playing:
                        channel.stop()
                        self.is_playing = False
                        
        except Exception as e:
            logger.error(f"音量更新エラー: {e}")
    
    def check_timeout(self):
        """検出タイムアウトのチェック"""
        try:
            if time.time() - self.last_detection_time > self.no_detection_timeout:
                channel = self.audio_manager.get_channel(self.channel_id)
                if channel:
                    channel.stop()
                self.current_volume = 0.0
                self.target_volume = 0.0
                self.motion_metrics = MotionMetrics()
                self.is_in_contact = False
                
        except Exception as e:
            logger.error(f"タイムアウトチェックエラー: {e}")
    
    def cleanup(self):
        """リソースの解放"""
        try:
            channel = self.audio_manager.get_channel(self.channel_id)
            if channel:
                channel.stop()
            
        except Exception as e:
            logger.error(f"SoundController クリーンアップエラー: {e}")

class PointCloudProcessor:
    """点群データの処理を行うクラス"""
    def __init__(self, params: SystemParameters, audio_manager: AudioManager):
        self.params = params
        self.audio_manager = audio_manager
        self.current_point_cloud = None
        self.kdtree = None
        self.processing_lock = threading.Lock()
        self.is_processing = False
        self.scan_progress = 0
        self.progress_callback = None
        
        # 点群処理パイプライン
        self.pipeline_stages = [
            self._remove_background,
            self._statistical_outlier_removal,
            self._estimate_normals,
            self._remove_invalid_normals,
            self._cluster_points,
            self._voxel_downsample
        ]
        
        # 処理用バッファ
        self.frame_buffer = deque(maxlen=params.FRAMES_TO_AVERAGE)
        
        # 音声効果の初期化
        self._initialize_sounds()
    
    def _initialize_sounds(self):
        """音声効果の初期化"""
        try:
            sound_files = {
                'start': 'music/A01.mp3',
                'processing': 'music/A02.mp3',
                'complete': 'music/A03.mp3'
            }
            
            for sound_id, file_path in sound_files.items():
                if not self.audio_manager.load_sound(f"scan_{sound_id}", file_path):
                    logger.warning(f"スキャン音声の読み込みに失敗: {sound_id}")
                    
        except Exception as e:
            logger.error(f"音声初期化エラー: {e}")

    def _create_point_cloud(self, points_list: List[np.ndarray]) -> Optional[o3d.geometry.PointCloud]:
        """点群オブジェクトの作成（改善版）"""
        try:
            if not points_list:
                return None
            
            # 全点群の結合
            all_points = np.vstack(points_list)
            
            if len(all_points) == 0:
                return None
            
            # Open3D点群オブジェクトの作成
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(all_points)
            
            if DEBUG_MODE:
                logger.debug(f"作成された点群の点数: {len(pcd.points)}")
            
            return pcd
            
        except Exception as e:
            logger.error(f"点群オブジェクト作成エラー: {e}")
            return None

    def set_progress_callback(self, callback):
        """進捗通知用のコールバックを設定"""
        self.progress_callback = callback
    
    def _notify_progress(self, progress: float, message: str = ""):
        """進捗の通知"""
        self.scan_progress = progress
        if self.progress_callback:
            self.progress_callback(progress, message)
        if DEBUG_MODE:
            logger.debug(f"Scan progress: {progress:.1f}% - {message}")

    def process_point_cloud(self, depth_frames: List[rs.frame]) -> Tuple[Optional[o3d.geometry.PointCloud], Optional[np.ndarray]]:
        """点群の処理メインルーチン（進捗表示付き）"""
        if self.is_processing:
            logger.warning("点群処理が既に実行中です")
            return None, None
        
        try:
            with self.processing_lock:
                self.is_processing = True
                logger.info("点群処理を開始します")
                self._notify_progress(0, "スキャン開始")
                
                # 点群の蓄積
                self._notify_progress(10, "フレームデータ収集中")
                accumulated_points = self._accumulate_points(depth_frames)
                if not accumulated_points:
                    raise RuntimeError("有効な点群データを取得できませんでした")
                
                # 点群オブジェクトの作成
                self._notify_progress(30, "点群データ生成中")
                pcd = self._create_point_cloud(accumulated_points)
                if pcd is None:
                    raise RuntimeError("点群オブジェクトの作成に失敗しました")
                
                # 前処理：重複点の除去
                self._notify_progress(50, "点群データ前処理中")
                pcd = pcd.voxel_down_sample(voxel_size=0.002)
                if len(pcd.points) == 0:
                    raise RuntimeError("ダウンサンプリング後に点群が空になりました")
                
                # 統計的外れ値除去
                self._notify_progress(60, "ノイズ除去中")
                pcd, _ = pcd.remove_statistical_outlier(
                    nb_neighbors=self.params.STATISTICAL_NB_NEIGHBORS,
                    std_ratio=self.params.STATISTICAL_STD_RATIO
                )
                
                # 法線の推定
                self._notify_progress(70, "表面法線を計算中")
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=self.params.NORMAL_RADIUS,
                        max_nn=self.params.NORMAL_MAX_NN
                    )
                )
                
                # 法線の向きを統一
                self._notify_progress(80, "法線方向を最適化中")
                pcd.orient_normals_towards_camera_location()
                
                # 最終的なダウンサンプリング
                self._notify_progress(90, "最終処理中")
                pcd = pcd.voxel_down_sample(voxel_size=self.params.VOXEL_SIZE)
                
                # 処理結果の保存
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"sandbox_processed_{timestamp}.ply"
                o3d.io.write_point_cloud(filename, pcd)
                logger.info(f"点群データを保存しました: {filename}")
                
                # KD木の更新
                self._notify_progress(95, "空間インデックスを構築中")
                self.current_point_cloud = pcd
                points_array = np.asarray(pcd.points)
                self.kdtree = KDTree(points_array)
                
                self._notify_progress(100, "スキャン完了")
                
                return pcd, points_array
                
        except Exception as e:
            logger.error(f"点群処理エラー: {str(e)}")
            self._notify_progress(-1, f"エラー: {str(e)}")
            return None, None
            
        finally:
            self.is_processing = False

    def _draw_scan_progress(self, image: np.ndarray):
        """スキャン進捗の描画"""
        try:
            if not self.is_processing and self.scan_progress == 0:
                return
            
            height, width = image.shape[:2]
            progress_width = int(width * 0.8)
            progress_height = 30
            x = int((width - progress_width) / 2)
            y = height - 50
            
            # 背景バー
            cv2.rectangle(
                image,
                (x, y),
                (x + progress_width, y + progress_height),
                (50, 50, 50),
                -1
            )
            
            # 進捗バー
            if self.scan_progress > 0:
                progress_x = int(progress_width * (self.scan_progress / 100))
                cv2.rectangle(
                    image,
                    (x, y),
                    (x + progress_x, y + progress_height),
                    (0, 255, 0) if self.scan_progress == 100 else (0, 165, 255),
                    -1
                )
            
            # 進捗テキスト
            text = f"スキャン進捗: {self.scan_progress:.1f}%"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = x + (progress_width - text_size[0]) // 2
            text_y = y + (progress_height + text_size[1]) // 2
            
            cv2.putText(
                image,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
        except Exception as e:
            logger.error(f"進捗描画エラー: {e}")

    def _debug_point_data(self, vertices):
        """点群データのデバッグ情報を出力"""
        try:
            if hasattr(vertices[0], 'x'):
                sample = vertices[0]
                logger.debug(f"頂点データ型: {type(sample)}")
                logger.debug(f"サンプル頂点: x={sample.x}, y={sample.y}, z={sample.z}")
            else:
                sample = vertices[0]
                logger.debug(f"頂点データ型: {type(sample)}")
                logger.debug(f"サンプル頂点: {sample}")
        except Exception as e:
            logger.debug(f"データ型デバッグエラー: {e}")

    def _accumulate_points(self, depth_frames: List[rs.frame]) -> List[np.ndarray]:
        """複数フレームからの点群蓄積（BufData処理の修正版）"""
        accumulated_points = []
        
        try:
            for depth_frame in depth_frames:
                if not depth_frame:
                    continue
                
                # 点群データの計算
                pc = rs.pointcloud()
                points = pc.calculate(depth_frame)
                
                try:
                    # バッファデータを numpy 配列に変換
                    vertices_buffer = np.asanyarray(points.get_vertices())
                    
                    if DEBUG_MODE:
                        logger.debug(f"Buffer shape: {vertices_buffer.shape}")
                        logger.debug(f"Buffer dtype: {vertices_buffer.dtype}")
                    
                    # 構造化配列から通常の配列に変換
                    vertices_array = np.vstack([
                        vertices_buffer['f0'],
                        vertices_buffer['f1'],
                        vertices_buffer['f2']
                    ]).T
                    
                    # データ型を float32 に変換
                    vertices_array = vertices_array.astype(np.float32)
                    
                    # 無効な点の除去
                    valid_mask = np.all(np.isfinite(vertices_array), axis=1)
                    valid_mask &= (vertices_array[:, 2] > 0) & (vertices_array[:, 2] < 2.0)
                    
                    if np.any(valid_mask):
                        valid_points = vertices_array[valid_mask]
                        
                        if len(valid_points) > 0:
                            accumulated_points.append(valid_points)
                            if DEBUG_MODE:
                                logger.debug(f"有効な点数: {len(valid_points)}")
                    
                except Exception as e:
                    logger.error(f"頂点データ変換エラー: {e}")
                    if DEBUG_MODE:
                        import traceback
                        logger.debug(traceback.format_exc())
                    continue
            
            if not accumulated_points:
                logger.warning("有効な点群データが得られませんでした")
                return []
            
            if DEBUG_MODE:
                total_points = sum(len(points) for points in accumulated_points)
                logger.debug(f"合計点数: {total_points}")
            
            return accumulated_points
            
        except Exception as e:
            logger.error(f"点群蓄積エラー: {str(e)}")
            if DEBUG_MODE:
                import traceback
                logger.debug(traceback.format_exc())
            return []
    
    def _remove_background(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """背景の除去"""
        try:
            points = np.asarray(pcd.points)
            
            # 高さに基づくフィルタリング
            height_mask = (points[:, 2] > 0.1) & (points[:, 2] < 1.5)
            
            # 水平面からの距離に基づくフィルタリング
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=0.02,
                ransac_n=3,
                num_iterations=1000
            )
            
            # フィルタリングの適用
            filtered_points = points[height_mask]
            filtered_pcd = o3d.geometry.PointCloud()
            filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
            
            return filtered_pcd
            
        except Exception as e:
            logger.error(f"背景除去エラー: {e}")
            return pcd
    
    def _statistical_outlier_removal(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """統計的外れ値の除去"""
        try:
            filtered_pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=self.params.STATISTICAL_NB_NEIGHBORS,
                std_ratio=self.params.STATISTICAL_STD_RATIO
            )
            return filtered_pcd
            
        except Exception as e:
            logger.error(f"外れ値除去エラー: {e}")
            return pcd
    
    def _estimate_normals(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """法線の推定"""
        try:
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.params.NORMAL_RADIUS,
                    max_nn=self.params.NORMAL_MAX_NN
                )
            )
            pcd.orient_normals_towards_camera_location()
            return pcd
            
        except Exception as e:
            logger.error(f"法線推定エラー: {e}")
            return pcd
    
    def _remove_invalid_normals(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """無効な法線を持つ点の除去"""
        try:
            normals = np.asarray(pcd.normals)
            points = np.asarray(pcd.points)
            
            # 法線の向きチェック
            up_vector = np.array([0, 0, 1])
            normal_angles = np.abs(np.dot(normals, up_vector))
            valid_normal_mask = normal_angles > 0.3
            
            if np.any(valid_normal_mask):
                filtered_pcd = o3d.geometry.PointCloud()
                filtered_pcd.points = o3d.utility.Vector3dVector(points[valid_normal_mask])
                filtered_pcd.normals = o3d.utility.Vector3dVector(normals[valid_normal_mask])
                return filtered_pcd
            
            return pcd
            
        except Exception as e:
            logger.error(f"法線フィルタリングエラー: {e}")
            return pcd
    
    def _cluster_points(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """点群のクラスタリング"""
        try:
            labels = np.array(pcd.cluster_dbscan(
                eps=self.params.DBSCAN_EPS,
                min_points=self.params.DBSCAN_MIN_POINTS
            ))
            
            if len(labels) > 0:
                unique_labels = np.unique(labels[labels >= 0])
                if len(unique_labels) > 0:
                    # 最大クラスタの抽出
                    largest_cluster_label = \
                        unique_labels[np.argmax([np.sum(labels == label) for label in unique_labels])]
                    largest_cluster_mask = labels == largest_cluster_label
                    
                    clustered_pcd = o3d.geometry.PointCloud()
                    clustered_pcd.points = o3d.utility.Vector3dVector(
                        np.asarray(pcd.points)[largest_cluster_mask]
                    )
                    if len(pcd.normals) > 0:
                        clustered_pcd.normals = o3d.utility.Vector3dVector(
                            np.asarray(pcd.normals)[largest_cluster_mask]
                        )
                    return clustered_pcd
            
            return pcd
            
        except Exception as e:
            logger.error(f"クラスタリングエラー: {e}")
            return pcd
    
    def _voxel_downsample(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """ボクセルダウンサンプリング"""
        try:
            return pcd.voxel_down_sample(voxel_size=self.params.VOXEL_SIZE)
        except Exception as e:
            logger.error(f"ダウンサンプリングエラー: {e}")
            return pcd
    
    def _update_spatial_index(self, pcd: o3d.geometry.PointCloud):
        """空間インデックスの更新"""
        try:
            self.current_point_cloud = pcd
            self.kdtree = KDTree(np.asarray(pcd.points))
        except Exception as e:
            logger.error(f"空間インデックス更新エラー: {e}")
    
    def get_nearest_distance(self, point: np.ndarray) -> Optional[float]:
        """最近傍点との距離を取得"""
        try:
            if self.kdtree is None:
                return None
                
            # 入力点の形状を確認
            if point.shape != (3,):
                point = point.flatten()[:3]
            
            # 無効な値のチェック
            if not np.all(np.isfinite(point)):
                return None
            
            distance, _ = self.kdtree.query(point)
            return float(distance)
            
        except Exception as e:
            logger.error(f"距離計算エラー: {str(e)}")
            return None
    
    def _play_sound(self, sound_type: str):
        """処理状態に応じた音声の再生"""
        try:
            channel = self.audio_manager.get_channel(15)  # 専用チャンネルを使用
            if channel:
                sound = self.audio_manager._sounds.get(f"scan_{sound_type}")
                if sound:
                    channel.play(sound)
        except Exception as e:
            logger.error(f"音声再生エラー: {e}")
    
    def cleanup(self):
        """リソースの解放"""
        try:
            with self.processing_lock:
                self.current_point_cloud = None
                self.kdtree = None
        except Exception as e:
            logger.error(f"PointCloudProcessor クリーンアップエラー: {str(e)}")

class SandboxSystem:
    """インタラクティブ砂場音響システムのメインクラス"""
    def __init__(self):
        self.params = SystemParameters()
        self.resource_manager = resource_manager  # グローバルなリソースマネージャー
        self.performance_monitor = PerformanceMonitor()
        self._initialize_system()
    
    def _initialize_system(self):
        """システムの初期化"""
        try:
            # システム要件のチェック
            success, error_message = initialize_system()
            if not success:
                raise RuntimeError(f"システム初期化エラー: {error_message}")
            
            # 音声システムの初期化
            self.audio_manager = AudioManager()
            if not self.audio_manager.initialize():
                raise RuntimeError("音声システムの初期化に失敗しました")
            self.resource_manager.register(self.audio_manager)
            
            # カメラシステムの初期化
            self.camera_manager = RealSenseManager(self.params)
            if not self.camera_manager.initialize():
                raise RuntimeError("カメラシステムの初期化に失敗しました")
            self.resource_manager.register(self.camera_manager)
            
            # 各コンポーネントの初期化
            self._initialize_components()
            
            # システム状態の初期化
            self.is_running = False
            self.consecutive_failures = 0
            self.last_reset_time = time.time()
            
            logger.info("システムの初期化が完了しました")
            
        except Exception as e:
            logger.error(f"システム初期化エラー: {e}")
            self.cleanup()
            raise
    
    def _initialize_components(self):
        """システムコンポーネントの初期化"""
        try:
            # 画像処理システムの初期化
            self.frame_processor = FrameProcessor(self.params)
            
            # ハンドトラッキングシステムの初期化
            self.hand_tracker = HandTracker(self.params)
            self.resource_manager.register(self.hand_tracker)
            
            # 点群処理システムの初期化
            self.point_cloud_processor = PointCloudProcessor(self.params, self.audio_manager)
            self.resource_manager.register(self.point_cloud_processor)
            
            # サウンドコントローラーの初期化
            self.sound_controllers = {
                "Right": SoundController(
                    self.audio_manager, self.params, "Right", channel_id=0
                ),
                "Left": SoundController(
                    self.audio_manager, self.params, "Left", channel_id=1
                )
            }
            for controller in self.sound_controllers.values():
                self.resource_manager.register(controller)
                
        except Exception as e:
            raise RuntimeError(f"コンポーネント初期化エラー: {e}")
    
    def run(self):
        """メインループ"""
        self.is_running = True
        logger.info("システムを開始します")
        
        try:
            while self.is_running:
                try:
                    # フレームレート監視の更新
                    self.performance_monitor.update()
                    
                    # 定期的なシステムメンテナンス
                    self._perform_maintenance()
                    
                    # フレームの取得と処理
                    if not self._process_frame():
                        continue
                    
                    # キー入力の処理
                    if not self._handle_input():
                        break
                    
                except Exception as e:
                    self._handle_frame_error(e)
                    
        except KeyboardInterrupt:
            logger.info("ユーザーによる中断を検出しました")
        except Exception as e:
            logger.error(f"メインループでエラーが発生: {e}")
        finally:
            self.cleanup()
    
    def _perform_maintenance(self):
        """システムメンテナンス処理"""
        try:
            current_time = time.time()
            
            # 定期的なリセットチェック
            if current_time - self.last_reset_time > self.params.processing.reset_interval:
                logger.info("定期的なシステムメンテナンスを実行します")
                self.camera_manager.reset()
                self.last_reset_time = current_time
                
            # パフォーマンス統計の記録
            if DEBUG_MODE:
                stats = self.performance_monitor.get_stats()
                logger.debug(
                    f"Performance: FPS={stats['fps']:.1f}, "
                    f"Frame time={stats['avg_ms']:.1f}ms"
                )
                
        except Exception as e:
            logger.error(f"メンテナンス処理エラー: {e}")
    
    def _process_frame(self) -> bool:
        """フレームの処理（非同期版）"""
        try:
            # フレームの取得
            depth_frame, color_frame = self.camera_manager.get_frames()
            if depth_frame is None or color_frame is None:
                self.consecutive_failures += 1
                return False
            
            self.consecutive_failures = 0
            
            # フレームの処理
            depth_image, color_image = self.frame_processor.process_frames(
                depth_frame, color_frame
            )
            
            # 手の検出と追跡
            hand_positions, hand_types, annotated_image = self.hand_tracker.process_frame(
                color_image, depth_frame, self.camera_manager.depth_scale
            )
            
            # 非同期で手の相互作用を処理
            threading.Thread(
                target=self._process_hand_interactions,
                args=(hand_positions, hand_types),
                daemon=True
            ).start()
            
            # 表示の更新
            self._update_display(annotated_image, depth_image)
            
            return True
            
        except Exception as e:
            logger.error(f"フレーム処理エラー: {e}")
            return False
    
    def _process_hand_interactions(self, hand_positions: List[np.ndarray], 
                                 hand_types: List[str]):
        """手の相互作用の処理"""
        try:
            if not hand_positions or self.point_cloud_processor.kdtree is None:
                return
            
            for hand_position, hand_type in zip(hand_positions, hand_types):
                if ((hand_type == "Right" and not self.params.DETECT_RIGHT_HAND) or
                    (hand_type == "Left" and not self.params.DETECT_LEFT_HAND)):
                    continue
                
                # 距離の計算
                distance = self.point_cloud_processor.get_nearest_distance(hand_position)
                if distance is not None:
                    # サウンドの更新
                    self.sound_controllers[hand_type].update_sound(
                        distance, hand_position
                    )
            
            # タイムアウトチェック
            for controller in self.sound_controllers.values():
                controller.check_timeout()
                
        except Exception as e:
            logger.error(f"手の相互作用処理エラー: {e}")
    
    def _update_display(self, color_image: np.ndarray, depth_image: np.ndarray):
        """ディスプレイの更新（進捗表示付き）"""
        try:
            # 深度画像のカラーマップ生成
            depth_colormap = self.frame_processor.create_depth_colormap(depth_image)
            
            # 画像サイズの確認とリサイズ
            color_h, color_w = color_image.shape[:2]
            depth_h, depth_w = depth_colormap.shape[:2]
            
            if color_h != depth_h or color_w != depth_w:
                depth_colormap = cv2.resize(
                    depth_colormap,
                    (color_w, color_h),
                    interpolation=cv2.INTER_AREA
                )
            
            # ステータス情報の描画
            self._draw_status(color_image)
            
            # スキャン進捗の描画
            self.point_cloud_processor._draw_scan_progress(color_image)
            
            # 画像の連結と表示
            display_image = np.hstack((color_image, depth_colormap))
            
            # ウィンドウサイズの調整
            window_width = 1280
            aspect_ratio = display_image.shape[1] / display_image.shape[0]
            window_height = int(window_width / aspect_ratio)
            
            display_image_resized = cv2.resize(
                display_image,
                (window_width, window_height),
                interpolation=cv2.INTER_AREA
            )
            
            cv2.imshow('Interactive Sandbox System', display_image_resized)
            
        except Exception as e:
            logger.error(f"ディスプレイ更新エラー: {e}")
            logger.debug(f"Color image shape: {color_image.shape}")
            logger.debug(f"Depth image shape: {depth_image.shape}")

    def create_depth_colormap(self, depth_image: np.ndarray) -> np.ndarray:
        """深度画像のカラーマップ作成（FrameProcessorクラス内）"""
        try:
            # 深度データの正規化
            depth_normalized = cv2.normalize(
                depth_image,
                None,
                0,
                255,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_8U
            )
            
            # カラーマップの適用
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            
            # 無効な深度値（0や非常に大きな値）の処理
            zero_mask = (depth_image == 0) | (depth_image > 5000)  # 5m以上は無効とする
            depth_colormap[zero_mask] = [0, 0, 0]  # 黒色で表示
            
            return depth_colormap
            
        except Exception as e:
            logger.error(f"深度カラーマップ作成エラー: {e}")
            return np.zeros_like(depth_image)

    # また、RealSenseManagerクラス内でのフレーム取得時にアライメントを確実に行う
    def get_frames(self) -> Tuple[Optional[rs.frame], Optional[rs.frame]]:
        """フレームの取得"""
        if not self._is_running:
            return None, None
        
        with self._lock:
            try:
                # フレームの取得
                frames = self.pipeline.wait_for_frames(
                    timeout_ms=self.params.processing.frame_timeout
                )
                if not frames:
                    raise RuntimeError("フレームの取得に失敗")
                
                # アライメントの適用
                aligned_frames = self.align.process(frames)
                
                # 深度フレームとカラーフレームの取得
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                # フレームの検証
                if not depth_frame or not color_frame:
                    raise RuntimeError("有効なフレームが取得できません")
                
                # フレームデータの検証
                if depth_frame.get_data_size() == 0 or color_frame.get_data_size() == 0:
                    raise RuntimeError("フレームデータが空です")
                
                return depth_frame, color_frame
                
            except Exception as e:
                logger.error(f"フレーム取得エラー: {e}")
                return None, None

    def _draw_status(self, image: np.ndarray):
        """ステータス情報の描画"""
        try:
            # FPS情報
            fps_text = f"FPS: {self.performance_monitor.get_fps():.1f}"
            cv2.putText(
                image, fps_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            
            # 検出状態
            status_text = (
                f"Right hand: {'ON' if self.params.DETECT_RIGHT_HAND else 'OFF'} | "
                f"Left hand: {'ON' if self.params.DETECT_LEFT_HAND else 'OFF'}"
            )
            cv2.putText(
                image, status_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            
            # コントロール説明
            help_texts = [
                "Controls:",
                "s: scan 3D surface",
                "r: toggle right hand",
                "l: toggle left hand",
                "q: quit"
            ]
            
            for i, text in enumerate(help_texts):
                cv2.putText(
                    image, text, (10, 90 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
                )
                
        except Exception as e:
            logger.error(f"ステータス描画エラー: {e}")
    
    def _handle_input(self) -> bool:
        """キー入力の処理"""
        try:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                logger.info("終了が要求されました")
                return False
                
            elif key == ord('s'):
                logger.info("3Dスキャンを開始します")
                frames = [self.camera_manager.get_frames()[0] 
                         for _ in range(self.params.FRAMES_TO_AVERAGE)]
                self.point_cloud_processor.process_point_cloud(frames)
                
            elif key == ord('r'):
                self.params.DETECT_RIGHT_HAND = not self.params.DETECT_RIGHT_HAND
                logger.info(f"右手の検出: {'ON' if self.params.DETECT_RIGHT_HAND else 'OFF'}")
                
            elif key == ord('l'):
                self.params.DETECT_LEFT_HAND = not self.params.DETECT_LEFT_HAND
                logger.info(f"左手の検出: {'ON' if self.params.DETECT_LEFT_HAND else 'OFF'}")
            
            return True
            
        except Exception as e:
            logger.error(f"入力処理エラー: {e}")
            return True
    
    def _handle_frame_error(self, error: Exception):
        """フレームエラーの処理"""
        self.consecutive_failures += 1
        
        if self.consecutive_failures >= self.params.processing.max_consecutive_failures:
            logger.error(f"連続エラーが発生しました: {error}")
            logger.info("システムの再起動を試みます")
            
            try:
                self.camera_manager.reset()
                self.consecutive_failures = 0
                self.last_reset_time = time.time()
            except Exception as reset_error:
                logger.error(f"システムリセットに失敗: {reset_error}")
    
    def cleanup(self):
        """システムのクリーンアップ"""
        try:
            logger.info("システムをシャットダウンします")
            self.is_running = False
            
            # すべてのリソースの解放
            self.resource_manager.cleanup_all()
            
            # ウィンドウの破棄
            cv2.destroyAllWindows()
            
            logger.info("システムのクリーンアップが完了しました")
            
        except Exception as e:
            logger.error(f"クリーンアップエラー: {e}")

def main():
    """メイン関数"""
    try:
        # ログファイルの初期化
        logging.info("インタラクティブ砂場音響システムを起動します")
        
        # システムの作成と実行
        system = SandboxSystem()
        system.run()
        
    except Exception as e:
        logging.error(f"システム実行エラー: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()