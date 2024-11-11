"""インタラクティブ砂場音響システム
コアコンポーネント
"""

import logging
import os
import sys
import time
import warnings
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from contextlib import contextmanager

# プロトバッフの警告を抑制
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')

# デバッグモードの設定
DEBUG_MODE = False

# ログディレクトリの作成
if not os.path.exists('logs'):
    os.makedirs('logs')

# ロギングの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sandbox_system.log'),
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
    depth_format: str = 'z16'
    color_format: str = 'bgr8'
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
        self.FRAMES_TO_AVERAGE = 10  # フレーム数を増やす
        self.STATISTICAL_NB_NEIGHBORS = 30  # 近傍点数を増やす
        self.STATISTICAL_STD_RATIO = 2.5  # 標準偏差の比率を緩和
        self.NORMAL_RADIUS = 0.05  # 法線計算の半径を大きく
        self.NORMAL_MAX_NN = 50  # 最大近傍点数を増やす
        self.DBSCAN_EPS = 0.03  # クラスタリングの距離閾値を大きく
        self.DBSCAN_MIN_POINTS = 5  # 最小点数を減らす
        self.VOXEL_SIZE = 0.01  # ボクセルサイズを大きく
        
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

class PerformanceMonitor:
    """パフォーマンス監視クラス"""
    def __init__(self, window_size: int = 100):
        self.frame_times = []
        self.last_time = time.perf_counter()
        self.fps = 0
        self._lock = threading.Lock()
        self.window_size = window_size
    
    def update(self):
        """フレーム時間の更新"""
        current_time = time.perf_counter()
        with self._lock:
            frame_time = current_time - self.last_time
            self.frame_times.append(frame_time)
            if len(self.frame_times) > self.window_size:
                self.frame_times.pop(0)
            
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

@contextmanager
def error_handler(operation_name: str):
    """エラーハンドリング用コンテキストマネージャ"""
    try:
        yield
    except Exception as e:
        logger.error(f"{operation_name}でエラーが発生: {e}", exc_info=True)
        raise

def check_system_requirements() -> Tuple[bool, str]:
    """システム要件のチェック"""
    try:
        messages = []
        
        # 必要なディレクトリの存在確認と作成
        required_dirs = ['music', 'logs']
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                try:
                    os.makedirs(dir_name)
                    messages.append(f"ディレクトリを作成: {dir_name}")
                except Exception as e:
                    return False, f"ディレクトリ作成エラー: {e}"

        # 必要な音声ファイルの確認
        required_files = [f'A0{i}.mp3' for i in range(8)]  # A00.mp3 から A07.mp3
        missing_files = []
        for filename in required_files:
            filepath = os.path.join('music', filename)
            if not os.path.exists(filepath):
                missing_files.append(filename)

        if missing_files:
            return False, (f"必要な音声ファイルが不足しています: {', '.join(missing_files)}\n"
                         f"music フォルダに以下のファイルを配置してください:\n"
                         f"- " + "\n- ".join(required_files))

        # PyGameの初期化チェック
        try:
            import pygame
            if not pygame.get_init():
                pygame.init()
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            messages.append("PyGameの初期化成功")
        except Exception as e:
            return False, f"PyGameの初期化エラー: {e}"

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

# グローバルなリソースマネージャーの作成
resource_manager = ResourceManager()