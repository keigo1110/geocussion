"""インタラクティブ砂場音響システム
改善版カメラ管理とフレーム処理コンポーネント
"""

import numpy as np
import pyrealsense2 as rs
import cv2
import time
import threading
from typing import Optional, Tuple, List, Dict
import logging
from .core import error_handler, SystemParameters, DEBUG_MODE

logger = logging.getLogger(__name__)

class RealSenseManager:
    """RealSenseカメラの管理クラス（改善版）"""
    def __init__(self, params: SystemParameters):
        self.params = params
        self.pipeline: Optional[rs.pipeline] = None
        self.config: Optional[rs.config] = None
        self.align: Optional[rs.align] = None
        self.device: Optional[rs.device] = None
        self.depth_scale: float = 0.001
        self._lock = threading.Lock()
        self._frame_count = 0
        self._is_running = False
        self._last_reset_time = time.time()
        self._supported_profiles: Dict = {}
        
    def _check_device_compatibility(self) -> bool:
        """デバイスの互換性チェック"""
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            
            if not devices:
                logger.error("RealSenseデバイスが見つかりません")
                return False
            
            self.device = devices[0]
            device_name = self.device.get_info(rs.camera_info.name)
            serial_number = self.device.get_info(rs.camera_info.serial_number)
            firmware_version = self.device.get_info(rs.camera_info.firmware_version)
            
            logger.info(f"デバイス検出: {device_name} (S/N: {serial_number}, FW: {firmware_version})")
            
            # 利用可能なストリームプロファイルの確認
            self._supported_profiles = self._get_supported_profiles()
            return True
            
        except Exception as e:
            logger.error(f"デバイス互換性チェックエラー: {e}")
            return False
    
    def _get_supported_profiles(self) -> Dict:
        """サポートされているストリームプロファイルの取得"""
        profiles = {'depth': [], 'color': []}
        
        try:
            for sensor in self.device.query_sensors():
                for profile in sensor.get_stream_profiles():
                    try:
                        video_profile = profile.as_video_stream_profile()
                        stream_type = 'depth' if profile.stream_type() == rs.stream.depth else 'color'
                        
                        profiles[stream_type].append({
                            'width': video_profile.width(),
                            'height': video_profile.height(),
                            'fps': profile.fps(),
                            'format': profile.format()
                        })
                    except Exception:
                        continue
                        
            logger.debug(f"利用可能なプロファイル: {profiles}")
            return profiles
            
        except Exception as e:
            logger.error(f"プロファイル取得エラー: {e}")
            return profiles

    def initialize(self) -> bool:
        """カメラシステムの初期化"""
        try:
            if not self._check_device_compatibility():
                return False
            
            # パイプラインとコンフィグの作成
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # デバイスの有効化
            self.config.enable_device(
                self.device.get_info(rs.camera_info.serial_number)
            )
            
            # ストリームの設定
            if not self._configure_streams():
                return False
            
            # パイプラインの開始
            logger.info("パイプラインを開始します...")
            self.pipeline_profile = self.pipeline.start(self.config)
            
            # デプススケールの取得と設定
            depth_sensor = self.pipeline_profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            logger.info(f"深度スケール: {self.depth_scale}")
            
            # センサーの最適化
            self._optimize_sensors(depth_sensor)
            
            # アライメントオブジェクトの作成
            self.align = rs.align(rs.stream.color)
            
            # ウォームアップ
            if not self._warmup():
                return False
            
            self._is_running = True
            logger.info("カメラシステムの初期化が完了しました")
            return True
            
        except Exception as e:
            logger.error(f"カメラの初期化に失敗しました: {e}")
            self.cleanup()
            return False
    
    def _configure_streams(self) -> bool:
        """ストリームの設定（改善版）"""
        try:
            logger.info("ストリーム設定を開始します...")
            
            # 目的の設定
            target_config = {
                'depth': {
                    'width': self.params.camera.width,
                    'height': self.params.camera.height,
                    'fps': self.params.camera.fps,
                    'format': rs.format.z16
                },
                'color': {
                    'width': self.params.camera.width,
                    'height': self.params.camera.height,
                    'fps': self.params.camera.fps,
                    'format': rs.format.bgr8
                }
            }
            
            # 最適なプロファイルの選択と設定
            for stream_type, target in target_config.items():
                if not self._configure_single_stream(stream_type, target):
                    return False
            
            logger.info("ストリーム設定が完了しました")
            return True
            
        except Exception as e:
            logger.error(f"ストリーム設定エラー: {e}")
            return False
    
    def _configure_single_stream(self, stream_type: str, target: Dict) -> bool:
        """単一ストリームの設定"""
        try:
            available_profiles = self._supported_profiles.get(stream_type, [])
            
            if not available_profiles:
                logger.error(f"{stream_type}ストリームのプロファイルが見つかりません")
                return False
            
            # 最適なプロファイルの検索
            best_profile = None
            min_diff = float('inf')
            
            for profile in available_profiles:
                diff = abs(profile['width'] - target['width']) + \
                       abs(profile['height'] - target['height']) + \
                       abs(profile['fps'] - target['fps'])
                
                if diff < min_diff:
                    min_diff = diff
                    best_profile = profile
            
            if best_profile is None:
                logger.error(f"{stream_type}ストリームの適切なプロファイルが見つかりません")
                return False
            
            # ストリームの有効化
            stream = rs.stream.depth if stream_type == 'depth' else rs.stream.color
            self.config.enable_stream(
                stream,
                best_profile['width'],
                best_profile['height'],
                best_profile['format'],
                best_profile['fps']
            )
            
            logger.info(f"{stream_type}ストリーム設定: "
                       f"{best_profile['width']}x{best_profile['height']} "
                       f"@{best_profile['fps']}fps")
            return True
            
        except Exception as e:
            logger.error(f"{stream_type}ストリーム設定エラー: {e}")
            return False

    def _optimize_sensors(self, depth_sensor: rs.sensor):
        """センサーの最適化（改善版）"""
        try:
            # プリセットの設定
            if depth_sensor.supports(rs.option.visual_preset):
                depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy
                logger.info("深度センサープリセット: High Accuracy")
            
            # 最適化設定
            optimization_settings = {
                rs.option.laser_power: 1.0,  # 最大値に対する比率
                rs.option.confidence_threshold: 3,
                rs.option.noise_filtering: 1,
                rs.option.post_processing_sharpening: 3,
                rs.option.pre_processing_sharpening: 5
            }
            
            # 各オプションの設定
            for option, target_value in optimization_settings.items():
                if depth_sensor.supports(option):
                    try:
                        opt_range = depth_sensor.get_option_range(option)
                        if option == rs.option.laser_power:
                            value = opt_range.max * target_value
                        else:
                            value = min(max(target_value, opt_range.min), opt_range.max)
                        
                        depth_sensor.set_option(option, value)
                        if DEBUG_MODE:
                            logger.debug(f"センサーオプション {option}: {value}")
                            
                    except Exception as e:
                        logger.warning(f"オプション設定警告 ({option}): {e}")
            
        except Exception as e:
            logger.warning(f"センサー最適化警告: {e}")

    def _warmup(self) -> bool:
        """ウォームアップ（改善版）"""
        try:
            logger.info(f"ウォームアップ開始 ({self.params.processing.warmup_frames}フレーム)")
            
            for i in range(self.params.processing.warmup_frames):
                try:
                    frames = self.pipeline.wait_for_frames(
                        timeout_ms=self.params.processing.frame_timeout
                    )
                    
                    if not frames:
                        logger.warning(f"ウォームアップフレーム{i+1}の取得に失敗")
                        continue
                        
                    if DEBUG_MODE and (i + 1) % 10 == 0:
                        logger.debug(f"ウォームアップ進捗: {i+1}/{self.params.processing.warmup_frames}")
                        
                except Exception as e:
                    logger.warning(f"ウォームアップフレーム{i+1}エラー: {e}")
                    continue
            
            logger.info("ウォームアップ完了")
            return True
            
        except Exception as e:
            logger.error(f"ウォームアップエラー: {e}")
            return False

    def get_frames(self) -> Tuple[Optional[rs.frame], Optional[rs.frame]]:
        """フレームの取得（改善版）"""
        if not self._is_running:
            return None, None

        with self._lock:
            try:
                # 定期的なリセットの確認
                current_time = time.time()
                if current_time - self._last_reset_time > self.params.processing.reset_interval:
                    logger.info("定期的なカメラリセットを実行します")
                    self.reset()
                    self._last_reset_time = current_time
                
                # フレームの取得
                frames = self.pipeline.wait_for_frames(
                    timeout_ms=self.params.processing.frame_timeout
                )
                
                if not frames:
                    raise RuntimeError("フレームの取得に失敗")
                
                # アライメントの適用
                aligned_frames = self.align.process(frames)
                
                # フレームの取得と検証
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not self._validate_frames(depth_frame, color_frame):
                    return None, None
                
                self._frame_count += 1
                
                if DEBUG_MODE and self._frame_count % 30 == 0:
                    logger.debug(f"フレーム取得: {self._frame_count}")
                
                return depth_frame, color_frame
                
            except Exception as e:
                logger.error(f"フレーム取得エラー: {e}")
                return None, None
    
    def _validate_frames(self, depth_frame: rs.frame, color_frame: rs.frame) -> bool:
        """フレームの検証"""
        try:
            if not depth_frame or not color_frame:
                logger.warning("無効なフレーム")
                return False
            
            if depth_frame.get_data_size() == 0 or color_frame.get_data_size() == 0:
                logger.warning("空のフレームデータ")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"フレーム検証エラー: {e}")
            return False

    def reset(self):
        """カメラシステムのリセット（改善版）"""
        with self._lock:
            try:
                logger.info("カメラリセットを開始します")
                self._is_running = False
                
                if self.pipeline:
                    self.pipeline.stop()
                    logger.info("パイプラインを停止しました")
                
                time.sleep(1)  # リセット待機
                
                success = self.initialize()
                if success:
                    logger.info("カメラリセットが完了しました")
                else:
                    logger.error("カメラリセットに失敗しました")
                
            except Exception as e:
                logger.error(f"カメラリセットエラー: {e}")

    def cleanup(self):
        """リソースの解放（改善版）"""
        with self._lock:
            try:
                logger.info("カメラリソースの解放を開始します")
                self._is_running = False
                
                if self.pipeline:
                    self.pipeline.stop()
                    logger.info("パイプラインを停止しました")
                
                self.pipeline = None
                self.config = None
                self.align = None
                self.device = None
                
                logger.info("カメラリソースを解放しました")
                
            except Exception as e:
                logger.error(f"カメラクリーンアップエラー: {e}")

class FrameProcessor:
    """フレーム処理クラス（改善版）"""
    def __init__(self, params: SystemParameters):
        self.params = params
        self.depth_processor = rs.decimation_filter()
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()
        self.hole_filling_filter = rs.hole_filling_filter()
        self._configure_filters()
        self._frame_count = 0
    
    def _configure_filters(self):
        """フィルターの設定"""
        try:
            # デシメーションフィルタの設定
            self.depth_processor.set_option(rs.option.filter_magnitude, 2)
            
            # スパシャルフィルターの設定
            self.spatial_filter.set_option(rs.option.filter_magnitude, 2)
            self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
            self.spatial_filter.set_option(rs.option.filter_smooth_delta, 20)

            # テンポラルフィルターの設定
            self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
            self.temporal_filter.set_option(rs.option.filter_smooth_delta, 20)

            logger.info("フレーム処理フィルターを設定しました")
        except Exception as e:
            logger.warning(f"フィルター設定警告: {e}")

    def process_frames(self, depth_frame: rs.frame, color_frame: rs.frame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            """フレームの処理（修正版）"""
            try:
                if depth_frame is None or color_frame is None:
                    return None, None
                
                # 深度フレームのフィルタリング
                filtered_depth = self._filter_depth_frame(depth_frame)
                
                # numpy配列への変換
                depth_image = np.asanyarray(filtered_depth.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # カラー画像が2次元の場合は3チャンネルに変換
                if len(color_image.shape) == 2:
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_GRAY2BGR)
                
                # 深度画像の後処理
                depth_image = self._process_depth_image(depth_image)
                
                # サイズの正規化
                if color_image.shape[:2] != (480, 640):
                    color_image = cv2.resize(color_image, (640, 480))
                if depth_image.shape[:2] != (480, 640):
                    depth_image = cv2.resize(depth_image, (640, 480))
                
                self._frame_count += 1
                if DEBUG_MODE and self._frame_count % 30 == 0:
                    logger.debug(f"フレーム処理カウント: {self._frame_count}")
                    logger.debug(f"color_image shape: {color_image.shape}")
                    logger.debug(f"depth_image shape: {depth_image.shape}")
                
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

    def _process_depth_image(self, depth_image: np.ndarray) -> np.ndarray:
        """深度画像の後処理"""
        try:
            # 無効な深度値の除去
            depth_image[depth_image <= 0] = 0
            depth_image[depth_image > 5000] = 0  # 5m以上は無効

            # メディアンフィルタでノイズ除去
            depth_image = cv2.medianBlur(depth_image.astype(np.float32), 5)
            
            # オプショナル: エッジ保持平滑化
            if DEBUG_MODE:
                depth_image = cv2.bilateralFilter(
                    depth_image, d=5, sigmaColor=50, sigmaSpace=50
                )

            return depth_image

        except Exception as e:
            logger.error(f"深度画像処理エラー: {e}")
            return depth_image

    def create_depth_colormap(self, depth_image: np.ndarray) -> np.ndarray:
            """深度画像のカラーマップ作成（修正版）"""
            try:
                if depth_image is None:
                    return np.zeros((480, 640, 3), dtype=np.uint8)

                # サイズの正規化（640x480にリサイズ）
                if depth_image.shape[:2] != (480, 640):
                    depth_image = cv2.resize(depth_image, (640, 480))

                # 深度値の範囲を正規化（16ビット→8ビット）
                depth_min = np.min(depth_image[depth_image > 0]) if np.any(depth_image > 0) else 0
                depth_max = np.max(depth_image)
                
                if depth_max - depth_min > 0:
                    depth_scaled = ((depth_image - depth_min) * 255 / (depth_max - depth_min))
                else:
                    depth_scaled = depth_image * 0
                
                depth_normalized = depth_scaled.astype(np.uint8)
                
                # カラーマップの適用
                depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                
                # 無効な深度値の処理
                invalid_mask = (depth_image == 0) | (depth_image > 5000)
                invalid_mask_resized = cv2.resize(invalid_mask.astype(np.uint8), (640, 480)) > 0
                depth_colormap[invalid_mask_resized] = [0, 0, 0]  # 黒色で表示
                
                if DEBUG_MODE:
                    # コントラスト強調
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    depth_colormap_lab = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2LAB)
                    depth_colormap_lab[:,:,0] = clahe.apply(depth_colormap_lab[:,:,0])
                    depth_colormap = cv2.cvtColor(depth_colormap_lab, cv2.COLOR_LAB2BGR)
                    
                    # デバッグ情報の表示
                    cv2.putText(
                        depth_colormap,
                        f"Range: {depth_min:.1f}-{depth_max:.1f}mm",
                        (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )
                    
                    # 画像サイズの表示
                    cv2.putText(
                        depth_colormap,
                        f"Size: {depth_colormap.shape[1]}x{depth_colormap.shape[0]}",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )
                
                return depth_colormap
                
            except Exception as e:
                logger.error(f"深度カラーマップ作成エラー: {e}")
                return np.zeros((480, 640, 3), dtype=np.uint8)