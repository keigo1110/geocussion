"""インタラクティブ砂場音響システム
メインシステムと統合コンポーネント
"""

import cv2
import numpy as np
import threading
import time
from typing import Optional, Tuple, List, Dict
import logging
import sys
from .core import (
    SystemParameters,
    resource_manager,
    PerformanceMonitor,
    initialize_system,
    DEBUG_MODE
)
from .camera import RealSenseManager, FrameProcessor
from .hand_tracking import HandTracker
from .audio import AudioManager, SoundController
from .pointcloud import PointCloudProcessor

logger = logging.getLogger(__name__)

class SandboxSystem:
    """インタラクティブ砂場音響システムのメインクラス"""
    def __init__(self):
        try:
            self.params = SystemParameters()
            self.resource_manager = resource_manager
            self.performance_monitor = PerformanceMonitor()
            
            # システム状態
            self._is_running = False
            self._lock = threading.Lock()
            self.consecutive_failures = 0
            self.last_reset_time = time.time()
            
            # システムの初期化
            self._initialize_system()
            
        except Exception as e:
            logger.error(f"システム初期化エラー: {e}")
            self.cleanup()
            raise RuntimeError(f"システム初期化エラー: {e}")
    
    def _initialize_system(self):
        """システムの初期化"""
        try:
            # システム要件のチェック
            success, error_message = initialize_system()
            if not success:
                raise RuntimeError(f"システム要件チェック失敗: {error_message}")
            
            # コンポーネントの初期化
            self._initialize_components()
            
            logger.info("システムの初期化が完了しました")
            
        except Exception as e:
            logger.error(f"システム初期化エラー: {e}")
            raise
    
    def _initialize_components(self):
        """システムコンポーネントの初期化"""
        try:
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
            
            # 画像処理システムの初期化
            self.frame_processor = FrameProcessor(self.params)
            
            # ハンドトラッキングシステムの初期化
            self.hand_tracker = HandTracker(self.params)
            self.resource_manager.register(self.hand_tracker)
            
            # 点群処理システムの初期化
            self.point_cloud_processor = PointCloudProcessor(self.params, self.audio_manager)
            self.resource_manager.register(self.point_cloud_processor)
            
            # サウンドコントローラーの初期化
            self._initialize_sound_controllers()
            
        except Exception as e:
            raise RuntimeError(f"コンポーネント初期化エラー: {e}")
    
    def _initialize_sound_controllers(self):
        """サウンドコントローラーの初期化（修正版）"""
        try:
            self.sound_controllers = {
                "Right": SoundController(
                    self.audio_manager,
                    self.params,
                    "Right",
                    channel_id=0
                ),
                "Left": SoundController(
                    self.audio_manager,
                    self.params,
                    "Left",
                    channel_id=1
                )
            }
            
            for controller in self.sound_controllers.values():
                self.resource_manager.register(controller)
                
        except Exception as e:
            logger.error(f"サウンドコントローラー初期化エラー: {e}")
            raise RuntimeError(f"サウンドコントローラー初期化エラー: {e}")

    def run(self):
        """メインループ"""
        try:
            self._is_running = True
            logger.info("システムを開始します")
            
            while self._is_running:
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
                with self._lock:
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
        """フレームの処理"""
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
            
            if depth_image is None or color_image is None:
                return False
            
            # 手の検出と追跡
            hand_positions, hand_types, annotated_image = self.hand_tracker.process_frame(
                color_image, depth_frame, self.camera_manager.depth_scale
            )
            
            # 非同期で手の相互作用を処理
            self._process_hand_interactions(hand_positions, hand_types)
            
            # 表示の更新
            self._update_display(annotated_image, depth_image)
            
            return True
            
        except Exception as e:
            logger.error(f"フレーム処理エラー: {e}")
            return False

    def _calculate_simple_distance(self, hand_position: np.ndarray) -> Optional[float]:
        """簡易的な距離計算"""
        try:
            # 基準面からの高さを距離として使用
            BASE_HEIGHT = 0.5  # 基準となる高さ（メートル）
            distance = abs(hand_position[2] - BASE_HEIGHT)
            return distance if 0 <= distance <= 1.0 else None
            
        except Exception as e:
            logger.error(f"距離計算エラー: {e}")
            return None

    def _perform_quick_scan(self):
        """初期化用の簡易スキャン"""
        try:
            # 1フレームだけ使用して簡易的な点群を生成
            depth_frame, _ = self.camera_manager.get_frames()
            if depth_frame is not None:
                self.point_cloud_processor.process_point_cloud([depth_frame])
                
        except Exception as e:
            logger.error(f"簡易スキャンエラー: {e}")

    def _process_hand_interactions(self, hand_positions: List[np.ndarray], 
                                 hand_types: List[str]):
        """手の相互作用の処理（改善版）"""
        try:
            if not hand_positions:
                for controller in self.sound_controllers.values():
                    controller.check_timeout()
                return

            # 初期スキャンがまだの場合は実行
            if self.point_cloud_processor.kdtree is None:
                self._perform_quick_scan()
                return

            # 各手の処理
            for hand_position, hand_type in zip(hand_positions, hand_types):
                if ((hand_type == "Right" and not self.params.DETECT_RIGHT_HAND) or
                    (hand_type == "Left" and not self.params.DETECT_LEFT_HAND)):
                    continue

                try:
                    # 簡易的な距離計算（直接的な高さの差分）
                    distance = self._calculate_simple_distance(hand_position)
                    
                    if distance is not None:
                        if DEBUG_MODE:
                            logger.debug(f"{hand_type} Hand - Position: {hand_position}, Distance: {distance:.3f}m")
                        
                        controller = self.sound_controllers[hand_type]
                        controller.update_sound(distance, hand_position)
                        controller.last_distance = distance
                        controller.last_detection_time = time.time()
                    
                except Exception as e:
                    logger.error(f"{hand_type} Hand の処理でエラー: {e}")

            # 未検出の手のタイムアウト処理
            detected_types = set(hand_types)
            for hand_type, controller in self.sound_controllers.items():
                if hand_type not in detected_types:
                    controller.check_timeout()

        except Exception as e:
            logger.error(f"手の相互作用処理エラー: {e}")
    
    def _update_display(self, color_image: np.ndarray, depth_image: np.ndarray):
            """ディスプレイの更新（修正版）"""
            try:
                if color_image is None or depth_image is None:
                    logger.warning("画像データがありません")
                    return
                
                # color_imageが2次元の場合、3チャンネルに変換
                if len(color_image.shape) == 2:
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_GRAY2BGR)
                
                logger.debug(f"処理後のcolor_image shape: {color_image.shape}")
                
                # 深度画像のカラーマップ生成
                depth_colormap = self.frame_processor.create_depth_colormap(depth_image)
                logger.debug(f"depth_colormap shape: {depth_colormap.shape}")
                
                # 画像サイズの確認
                color_h, color_w = color_image.shape[:2]
                depth_h, depth_w = depth_colormap.shape[:2]
                
                # リサイズが必要な場合
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
                
                # 画像の連結
                try:
                    display_image = np.hstack((color_image, depth_colormap))
                except Exception as e:
                    logger.error(f"画像連結エラー: {e}")
                    logger.debug(f"最終color_image shape: {color_image.shape}")
                    logger.debug(f"最終depth_colormap shape: {depth_colormap.shape}")
                    return
                
                # ウィンドウサイズの調整
                window_width = 1280
                window_height = int(window_width * color_h / (color_w * 2))
                
                display_image_resized = cv2.resize(
                    display_image,
                    (window_width, window_height),
                    interpolation=cv2.INTER_AREA
                )
                
                # FPSの表示
                fps = self.performance_monitor.get_fps()
                cv2.putText(
                    display_image_resized,
                    f"FPS: {fps:.1f}",
                    (window_width - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )
                
                cv2.imshow('Interactive Sandbox System', display_image_resized)
                
            except Exception as e:
                logger.error(f"ディスプレイ更新エラー: {e}")
                if DEBUG_MODE:
                    import traceback
                    logger.debug(traceback.format_exc())
    
    def _draw_status(self, image: np.ndarray):
            """ステータス情報の描画（改善版）"""
            try:
                # 基本情報（白色）
                cv2.putText(
                    image,
                    f"FPS: {self.performance_monitor.get_fps():.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
                
                # 検出状態（緑/赤）
                hand_status = {
                    "Right": self.params.DETECT_RIGHT_HAND,
                    "Left": self.params.DETECT_LEFT_HAND
                }
                
                for i, (hand, enabled) in enumerate(hand_status.items()):
                    color = (0, 255, 0) if enabled else (0, 0, 255)
                    status = "ON" if enabled else "OFF"
                    cv2.putText(
                        image,
                        f"{hand}: {status}",
                        (10, 60 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )
                
                # 接触情報の表示
                y_offset = 120
                for hand_type, controller in self.sound_controllers.items():
                    if controller.is_in_contact:
                        cv2.putText(
                            image,
                            f"{hand_type} Contact! Distance: {controller.last_distance:.3f}m",
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),  # 黄色
                            2
                        )
                        if hasattr(controller, 'current_volume'):
                            cv2.putText(
                                image,
                                f"Volume: {controller.current_volume:.2f}",
                                (10, y_offset + 25),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 255),
                                2
                            )
                    y_offset += 60
                
                # コントロール説明
                controls = [
                    "Controls:",
                    "s: scan 3D surface",
                    "r: toggle right hand",
                    "l: toggle left hand",
                    "d: toggle debug info",
                    "q: quit"
                ]
                
                for i, text in enumerate(controls):
                    cv2.putText(
                        image,
                        text,
                        (10, y_offset + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1
                    )
                    
            except Exception as e:
                logger.error(f"ステータス描画エラー: {e}")
    
    def _handle_input(self) -> bool:
        """キー入力の処理"""
        try:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                logger.info("終了が要求されました")
                self._is_running = False
                return False
                
            elif key == ord('s'):
                logger.info("3Dスキャンを開始します")
                self._perform_3d_scan()
                
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
    
    def _perform_3d_scan(self):
        """詳細な3Dスキャン（非同期処理）"""
        def scan_thread():
            try:
                logger.info("詳細な3Dスキャンを開始します")
                frames = []
                for i in range(self.params.FRAMES_TO_AVERAGE):
                    depth_frame, _ = self.camera_manager.get_frames()
                    if depth_frame is not None:
                        frames.append(depth_frame)
                        if DEBUG_MODE and (i + 1) % 5 == 0:
                            logger.debug(f"フレーム収集進捗: {i + 1}/{self.params.FRAMES_TO_AVERAGE}")
                
                if frames:
                    pcd, points = self.point_cloud_processor.process_point_cloud(frames)
                    if pcd is not None:
                        logger.info(f"詳細スキャン完了: {len(points)} 点を検出")
                        # スキャンデータの保存
                        self.point_cloud_processor._save_point_cloud(pcd)
                    else:
                        logger.error("点群の処理に失敗しました")
                else:
                    logger.error("スキャン用のフレームを取得できませんでした")
                
            except Exception as e:
                logger.error(f"詳細スキャンエラー: {e}")

        # 非同期でスキャンを実行
        threading.Thread(target=scan_thread, daemon=True).start()
    
    def _handle_frame_error(self, error: Exception):
        """フレームエラーの処理"""
        self.consecutive_failures += 1
        
        if self.consecutive_failures >= self.params.processing.max_consecutive_failures:
            logger.error(f"連続エラーが発生しました: {error}")
            logger.info("システムの再起動を試みます")
            
            try:
                with self._lock:
                    self.camera_manager.reset()
                    self.consecutive_failures = 0
                    self.last_reset_time = time.time()
            except Exception as reset_error:
                logger.error(f"システムリセットに失敗: {reset_error}")
    
    def cleanup(self):
        """システムのクリーンアップ"""
        try:
            logger.info("システムをシャットダウンします")
            self._is_running = False
            
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
        return 1
    
    return 0

def run_with_error_handling():
    """エラーハンドリング付きでシステムを実行"""
    try:
        logger.info("システムを起動します")
        exit_code = main()
        logger.info(f"システムを終了します (終了コード: {exit_code})")
        return exit_code
        
    except KeyboardInterrupt:
        logger.info("ユーザーによる中断を検出しました")
        return 0
        
    except Exception as e:
        logger.critical(f"予期しないエラーが発生しました: {e}", exc_info=True)
        return 1
    
    finally:
        # 最終的なクリーンアップ
        try:
            cv2.destroyAllWindows()
            logger.info("すべてのウィンドウを閉じました")
        except Exception as e:
            logger.error(f"最終クリーンアップエラー: {e}")

class SystemMonitor:
    """システムの状態監視クラス"""
    def __init__(self, system: SandboxSystem):
        self.system = system
        self.start_time = time.time()
        self.last_check_time = self.start_time
        self.check_interval = 5.0  # 5秒ごとにチェック
        self.status_history = []
        
    def check_system_status(self) -> Dict:
        """システムの状態をチェック"""
        current_time = time.time()
        if current_time - self.last_check_time < self.check_interval:
            return {}
        
        try:
            status = {
                'timestamp': current_time,
                'uptime': current_time - self.start_time,
                'fps': self.system.performance_monitor.get_fps(),
                'performance': self.system.performance_monitor.get_stats(),
                'camera_status': self.system.camera_manager._is_running,
                'consecutive_failures': self.system.consecutive_failures,
                'memory_usage': self._get_memory_usage()
            }
            
            self.status_history.append(status)
            if len(self.status_history) > 100:
                self.status_history.pop(0)
            
            self.last_check_time = current_time
            return status
            
        except Exception as e:
            logger.error(f"システム状態チェックエラー: {e}")
            return {}

    def _get_memory_usage(self) -> Dict:
        """メモリ使用状況の取得"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss': memory_info.rss / (1024 * 1024),  # MB
                'vms': memory_info.vms / (1024 * 1024),  # MB
                'percent': process.memory_percent()
            }
        except Exception as e:
            logger.error(f"メモリ使用状況取得エラー: {e}")
            return {}

    def get_performance_summary(self) -> Dict:
        """パフォーマンスサマリーの取得"""
        try:
            if not self.status_history:
                return {}
            
            fps_values = [status['fps'] for status in self.status_history]
            failure_counts = [status['consecutive_failures'] 
                            for status in self.status_history]
            
            return {
                'average_fps': np.mean(fps_values),
                'min_fps': np.min(fps_values),
                'max_fps': np.max(fps_values),
                'fps_stability': np.std(fps_values),
                'total_failures': np.sum(failure_counts),
                'max_consecutive_failures': np.max(failure_counts)
            }
            
        except Exception as e:
            logger.error(f"パフォーマンスサマリー作成エラー: {e}")
            return {}

    def generate_report(self) -> str:
        """システム状態レポートの生成"""
        try:
            summary = self.get_performance_summary()
            current_status = self.check_system_status()
            
            report = [
                "システム状態レポート",
                "=" * 50,
                f"実行時間: {current_status.get('uptime', 0):.1f} 秒",
                f"現在のFPS: {current_status.get('fps', 0):.1f}",
                f"平均FPS: {summary.get('average_fps', 0):.1f}",
                f"FPS安定性: {summary.get('fps_stability', 0):.2f}",
                f"総エラー数: {summary.get('total_failures', 0)}",
                f"最大連続エラー: {summary.get('max_consecutive_failures', 0)}",
                "=" * 50
            ]
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"レポート生成エラー: {e}")
            return "レポート生成に失敗しました"

if __name__ == "__main__":
    sys.exit(run_with_error_handling())