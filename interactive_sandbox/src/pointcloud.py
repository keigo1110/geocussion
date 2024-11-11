"""インタラクティブ砂場音響システム
点群処理コンポーネント
"""

import numpy as np
import open3d as o3d
import threading
import time
from typing import Optional, Tuple, List, Dict
from scipy.spatial import KDTree
import logging
import pyrealsense2 as rs
import cv2
from .core import SystemParameters, DEBUG_MODE

logger = logging.getLogger(__name__)

class PointCloudProcessor:
    """点群データの処理を行うクラス"""
    def __init__(self, params: SystemParameters, audio_manager):
        self.params = params
        self.audio_manager = audio_manager
        self.current_point_cloud = None
        self.kdtree = None
        self.processing_lock = threading.Lock()
        self.is_processing = False
        self.scan_progress = 0
        self.progress_callback = None
        self.last_update_time = time.time()
        
        # 点群処理パイプライン
        self.pipeline_stages = [
            ('背景除去', self._remove_background),
            ('ノイズ除去', self._statistical_outlier_removal),
            ('法線推定', self._estimate_normals),
            ('法線検証', self._remove_invalid_normals),
            ('クラスタリング', self._cluster_points),
            ('ダウンサンプリング', self._voxel_downsample)
        ]
        
        # 処理用バッファ
        self.frame_buffer = []
        self.max_buffer_size = params.FRAMES_TO_AVERAGE
        
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
        """点群の処理メインルーチン"""
        if self.is_processing:
            logger.warning("点群処理が既に実行中です")
            return None, None
        
        try:
            with self.processing_lock:
                self.is_processing = True
                self._play_sound('start')
                logger.info("点群処理を開始します")
                self._notify_progress(0, "スキャン開始")
                
                # 点群の蓄積
                self._notify_progress(10, "フレームデータ収集中")
                self._play_sound('processing')
                accumulated_points = self._accumulate_points(depth_frames)
                if not accumulated_points:
                    raise RuntimeError("有効な点群データを取得できませんでした")
                
                # 点群オブジェクトの作成
                self._notify_progress(30, "点群データ生成中")
                pcd = self._create_point_cloud(accumulated_points)
                if pcd is None:
                    raise RuntimeError("点群オブジェクトの作成に失敗しました")
                
                # 点群処理パイプラインの実行
                total_stages = len(self.pipeline_stages)
                progress_per_stage = 50 / total_stages
                current_progress = 40
                
                for stage_name, stage_func in self.pipeline_stages:
                    self._notify_progress(current_progress, f"{stage_name}中...")
                    pcd = stage_func(pcd)
                    if pcd is None:
                        raise RuntimeError(f"{stage_name}に失敗しました")
                    current_progress += progress_per_stage
                
                # 点群の検証
                if len(pcd.points) == 0:
                    raise RuntimeError("処理後の点群が空です")
                
                # 処理結果の保存
                self._save_point_cloud(pcd)
                
                # KD木の更新
                self._notify_progress(95, "空間インデックスを構築中")
                points_array = self._update_spatial_index(pcd)
                
                self._notify_progress(100, "スキャン完了")
                self._play_sound('complete')
                
                return pcd, points_array
                
        except Exception as e:
            logger.error(f"点群処理エラー: {str(e)}")
            self._notify_progress(-1, f"エラー: {str(e)}")
            return None, None
            
        finally:
            self.is_processing = False

    def _create_point_cloud(self, points_list: List[np.ndarray]) -> Optional[o3d.geometry.PointCloud]:
        """点群オブジェクトの作成"""
        try:
            if not points_list:
                return None
            
            # 全点群の結合
            all_points = np.vstack(points_list)
            
            if len(all_points) == 0:
                return None
            
            # 異常値の除去
            valid_mask = np.all(np.isfinite(all_points), axis=1)
            valid_points = all_points[valid_mask]
            
            if len(valid_points) == 0:
                return None
            
            # Open3D点群オブジェクトの作成
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(valid_points)
            
            if DEBUG_MODE:
                logger.debug(f"作成された点群の点数: {len(pcd.points)}")
            
            return pcd
            
        except Exception as e:
            logger.error(f"点群オブジェクト作成エラー: {e}")
            return None

    def _accumulate_points(self, depth_frames: List[rs.frame]) -> List[np.ndarray]:
        """複数フレームからの点群蓄積"""
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
                    vertices = np.asanyarray(points.get_vertices())
                    vertices = np.array([(v[0], v[1], v[2]) for v in vertices])
                    
                    # 無効な点の除去
                    valid_mask = np.all(np.isfinite(vertices), axis=1)
                    valid_mask &= (vertices[:, 2] > 0) & (vertices[:, 2] < 2.0)
                    
                    if np.any(valid_mask):
                        valid_points = vertices[valid_mask]
                        accumulated_points.append(valid_points)
                    
                except Exception as e:
                    logger.error(f"点群変換エラー: {e}")
                    continue
            
            if not accumulated_points:
                logger.warning("有効な点群データが得られませんでした")
                return []
            
            return accumulated_points
            
        except Exception as e:
            logger.error(f"点群蓄積エラー: {str(e)}")
            return []

    def _remove_background(self, pcd: o3d.geometry.PointCloud) -> Optional[o3d.geometry.PointCloud]:
        """背景の除去"""
        try:
            points = np.asarray(pcd.points)
            
            # 高さに基づくフィルタリング
            height_mask = (points[:, 2] > 0.1) & (points[:, 2] < 1.5)
            
            # 平面検出と除去
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=0.02,
                ransac_n=3,
                num_iterations=1000
            )
            
            # 平面以外の点を抽出
            outlier_cloud = pcd.select_by_index(inliers, invert=True)
            
            # 高さフィルタの適用
            filtered_points = np.asarray(outlier_cloud.points)[height_mask]
            
            if len(filtered_points) == 0:
                return None
            
            filtered_pcd = o3d.geometry.PointCloud()
            filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
            
            return filtered_pcd
            
        except Exception as e:
            logger.error(f"背景除去エラー: {e}")
            return None

    def _statistical_outlier_removal(self, pcd: o3d.geometry.PointCloud) -> Optional[o3d.geometry.PointCloud]:
        """統計的外れ値の除去"""
        try:
            if len(pcd.points) < self.params.STATISTICAL_NB_NEIGHBORS:
                return pcd
            
            filtered_pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=self.params.STATISTICAL_NB_NEIGHBORS,
                std_ratio=self.params.STATISTICAL_STD_RATIO
            )
            
            return filtered_pcd if len(filtered_pcd.points) > 0 else None
            
        except Exception as e:
            logger.error(f"外れ値除去エラー: {e}")
            return None

    def _estimate_normals(self, pcd: o3d.geometry.PointCloud) -> Optional[o3d.geometry.PointCloud]:
        """法線の推定"""
        try:
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.params.NORMAL_RADIUS,
                    max_nn=self.params.NORMAL_MAX_NN
                )
            )
            
            # 法線の向きを統一
            pcd.orient_normals_consistent_tangent_plane(100)
            return pcd
            
        except Exception as e:
            logger.error(f"法線推定エラー: {e}")
            return None

    def _remove_invalid_normals(self, pcd: o3d.geometry.PointCloud) -> Optional[o3d.geometry.PointCloud]:
        """無効な法線を持つ点の除去"""
        try:
            if not pcd.has_normals():
                return pcd
                
            normals = np.asarray(pcd.normals)
            points = np.asarray(pcd.points)
            
            # 法線の妥当性チェック
            normal_lengths = np.linalg.norm(normals, axis=1)
            valid_normal_mask = (normal_lengths > 0.1) & np.all(np.isfinite(normals), axis=1)
            
            if not np.any(valid_normal_mask):
                return pcd
            
            filtered_pcd = o3d.geometry.PointCloud()
            filtered_pcd.points = o3d.utility.Vector3dVector(points[valid_normal_mask])
            filtered_pcd.normals = o3d.utility.Vector3dVector(normals[valid_normal_mask])
            
            return filtered_pcd
            
        except Exception as e:
            logger.error(f"法線フィルタリングエラー: {e}")
            return None

    def _cluster_points(self, pcd: o3d.geometry.PointCloud) -> Optional[o3d.geometry.PointCloud]:
        """点群のクラスタリング"""
        try:
            if len(pcd.points) < self.params.DBSCAN_MIN_POINTS:
                return pcd
                
            # DBSCANクラスタリング
            labels = np.array(pcd.cluster_dbscan(
                eps=self.params.DBSCAN_EPS,
                min_points=self.params.DBSCAN_MIN_POINTS
            ))
            
            if len(labels) == 0:
                return pcd
            
            # 最大クラスタの抽出
            unique_labels = np.unique(labels[labels >= 0])
            if len(unique_labels) == 0:
                return pcd
                
            cluster_sizes = [np.sum(labels == label) for label in unique_labels]
            largest_cluster_label = unique_labels[np.argmax(cluster_sizes)]
            largest_cluster_mask = labels == largest_cluster_label
            
            clustered_pcd = o3d.geometry.PointCloud()
            clustered_pcd.points = o3d.utility.Vector3dVector(
                np.asarray(pcd.points)[largest_cluster_mask]
            )
            if pcd.has_normals():
                clustered_pcd.normals = o3d.utility.Vector3dVector(
                    np.asarray(pcd.normals)[largest_cluster_mask]
                )
                
            return clustered_pcd
            
        except Exception as e:
            logger.error(f"クラスタリングエラー: {e}")
            return None

    def _voxel_downsample(self, pcd: o3d.geometry.PointCloud) -> Optional[o3d.geometry.PointCloud]:
        """ボクセルダウンサンプリング"""
        try:
            return pcd.voxel_down_sample(voxel_size=self.params.VOXEL_SIZE)
        except Exception as e:
            logger.error(f"ダウンサンプリングエラー: {e}")
            return None

    def _save_point_cloud(self, pcd: o3d.geometry.PointCloud):
        """点群データの保存"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"sandbox_processed_{timestamp}.ply"
            o3d.io.write_point_cloud(filename, pcd)
            logger.info(f"点群データを保存しました: {filename}")
        except Exception as e:
            logger.error(f"点群保存エラー: {e}")

    def _update_spatial_index(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        """空間インデックスの更新"""
        try:
            self.current_point_cloud = pcd
            points_array = np.asarray(pcd.points)
            self.kdtree = KDTree(points_array)
            return points_array
            
        except Exception as e:
            logger.error(f"空間インデックス更新エラー: {e}")
            return None

    def get_nearest_distance(self, point: np.ndarray) -> Optional[float]:
        """最近傍点との距離を取得"""
        try:
            if self.kdtree is None:
                return None
            
            # 入力点の検証
            if not isinstance(point, np.ndarray):
                point = np.array(point)
            
            if point.shape != (3,):
                point = point.flatten()[:3]
            
            # 無効な値のチェック
            if not np.all(np.isfinite(point)):
                return None
            
            # 最近傍点の検索
            with self.processing_lock:
                distance, _ = self.kdtree.query(point)
                
                # 距離の妥当性チェック
                if not np.isfinite(distance) or distance < 0:
                    return None
                
                return float(distance)
            
        except Exception as e:
            logger.error(f"距離計算エラー: {str(e)}")
            return None

    def _play_sound(self, sound_type: str):
        """処理状態に応じた音声の再生"""
        try:
            channel = self.audio_manager.get_channel(15)  # 専用チャンネル使用
            if channel:
                sound = self.audio_manager._sounds.get(f"scan_{sound_type}")
                if sound:
                    channel.play(sound)
                    logger.debug(f"スキャン音声再生: {sound_type}")
        except Exception as e:
            logger.error(f"音声再生エラー: {e}")

    def analyze_surface(self, pcd: o3d.geometry.PointCloud) -> Dict:
        """表面特性の分析"""
        try:
            analysis_results = {
                'total_points': len(pcd.points),
                'surface_area': 0.0,
                'average_curvature': 0.0,
                'roughness': 0.0,
                'height_range': {'min': 0.0, 'max': 0.0}
            }
            
            points = np.asarray(pcd.points)
            if len(points) == 0:
                return analysis_results
            
            # 高さ範囲の計算
            height_values = points[:, 2]
            analysis_results['height_range'] = {
                'min': float(np.min(height_values)),
                'max': float(np.max(height_values))
            }
            
            # 表面積の概算
            if pcd.has_normals():
                normals = np.asarray(pcd.normals)
                area_elements = np.cross(points[:-1] - points[1:], normals[:-1])
                analysis_results['surface_area'] = float(np.sum(np.linalg.norm(area_elements, axis=1)) / 2)
            
            # 曲率と粗さの計算
            if len(points) >= 3:
                curvatures = []
                local_roughness = []
                
                for i in range(len(points)):
                    # 近傍点の取得
                    distances, indices = self.kdtree.query(points[i], k=min(10, len(points)))
                    if len(indices) >= 3:
                        # 局所的な曲率の計算
                        neighbors = points[indices]
                        centered = neighbors - neighbors.mean(axis=0)
                        _, s, _ = np.linalg.svd(centered)
                        curvature = s[2] / (s[0] + s[1] + s[2])
                        curvatures.append(curvature)
                        
                        # 局所的な粗さの計算
                        roughness = np.std(distances[1:])  # 最近傍点自身を除外
                        local_roughness.append(roughness)
                
                if curvatures:
                    analysis_results['average_curvature'] = float(np.mean(curvatures))
                if local_roughness:
                    analysis_results['roughness'] = float(np.mean(local_roughness))
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"表面分析エラー: {e}")
            return {}

    def detect_features(self, pcd: o3d.geometry.PointCloud) -> Dict:
        """特徴的な形状の検出"""
        try:
            features = {
                'peaks': [],
                'valleys': [],
                'steep_regions': []
            }
            
            points = np.asarray(pcd.points)
            if len(points) == 0:
                return features
            
            # 法線の計算（まだ無い場合）
            if not pcd.has_normals():
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=self.params.NORMAL_RADIUS,
                        max_nn=self.params.NORMAL_MAX_NN
                    )
                )
            
            normals = np.asarray(pcd.normals)
            
            # 特徴検出
            for i in range(len(points)):
                # 近傍点の取得
                distances, indices = self.kdtree.query(points[i], k=min(20, len(points)))
                if len(indices) >= 5:
                    neighbors = points[indices]
                    neighbor_normals = normals[indices]
                    
                    # 高さの相対的な位置
                    local_heights = neighbors[:, 2]
                    center_height = points[i, 2]
                    height_diff = center_height - np.mean(local_heights)
                    
                    # 法線の変化
                    normal_angles = np.abs(np.dot(neighbor_normals, normals[i]))
                    normal_variance = np.std(normal_angles)
                    
                    # ピークの検出
                    if height_diff > 0.02 and np.all(center_height >= local_heights[1:]):
                        features['peaks'].append({
                            'position': points[i].tolist(),
                            'prominence': float(height_diff)
                        })
                    
                    # 谷の検出
                    elif height_diff < -0.02 and np.all(center_height <= local_heights[1:]):
                        features['valleys'].append({
                            'position': points[i].tolist(),
                            'depth': float(-height_diff)
                        })
                    
                    # 急勾配領域の検出
                    if normal_variance > 0.3:
                        features['steep_regions'].append({
                            'position': points[i].tolist(),
                            'steepness': float(normal_variance)
                        })
            
            return features
            
        except Exception as e:
            logger.error(f"特徴検出エラー: {e}")
            return {'peaks': [], 'valleys': [], 'steep_regions': []}

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

    def cleanup(self):
        """リソースの解放"""
        try:
            with self.processing_lock:
                self.current_point_cloud = None
                self.kdtree = None
                self.is_processing = False
                self.frame_buffer.clear()
                logger.info("PointCloudProcessorのリソースを解放しました")
        except Exception as e:
            logger.error(f"PointCloudProcessor クリーンアップエラー: {str(e)}")