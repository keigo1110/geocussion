"""インタラクティブ砂場音響システム
音声管理と効果音制御コンポーネント
"""

import pygame
import os
import time
import threading
import numpy as np
from typing import Optional, Dict, Tuple
import logging
from .core import SystemParameters, DEBUG_MODE
from .hand_tracking import MotionMetrics

logger = logging.getLogger(__name__)

class AudioManager:
    """音声システム全体の管理クラス"""
    def __init__(self):
        self._lock = threading.Lock()
        self._initialized = False
        self._channels: Dict[int, pygame.mixer.Channel] = {}
        self._sounds: Dict[str, pygame.mixer.Sound] = {}
        self._volume_fade_threads: Dict[int, threading.Thread] = {}
        self._channel_states: Dict[int, Dict] = {}
    
    def initialize(self) -> bool:
        """音声システムの初期化"""
        try:
            with self._lock:
                if self._initialized:
                    return True
                
                # PyGameミキサーの初期化
                pygame.mixer.init(
                    frequency=44100,
                    size=-16,
                    channels=2,
                    buffer=512
                )
                
                # チャンネル数の設定
                pygame.mixer.set_num_channels(16)
                
                # チャンネル状態の初期化
                for i in range(16):
                    self._channel_states[i] = {
                        'volume': 0.0,
                        'target_volume': 0.0,
                        'fade_active': False
                    }
                
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
                if not self._initialized:
                    raise RuntimeError("音声システムが初期化されていません")
                
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"音声ファイルが見つかりません: {file_path}")
                
                try:
                    sound = pygame.mixer.Sound(file_path)
                    self._sounds[sound_id] = sound
                    logger.debug(f"音声ファイルを読み込みました: {sound_id}")
                    return True
                except Exception as e:
                    logger.error(f"音声ファイル読み込みエラー: {file_path} - {e}")
                    return False
                
        except Exception as e:
            logger.error(f"音声ファイルの読み込みに失敗 ({sound_id}): {e}")
            return False
    
    def get_channel(self, channel_id: int) -> Optional[pygame.mixer.Channel]:
        """専用チャンネルの取得"""
        try:
            with self._lock:
                if not self._initialized:
                    raise RuntimeError("音声システムが初期化されていません")
                
                if channel_id not in self._channels:
                    channel = pygame.mixer.Channel(channel_id)
                    self._channels[channel_id] = channel
                    # チャンネル状態の初期化
                    self._channel_states[channel_id] = {
                        'volume': 0.0,
                        'target_volume': 0.0,
                        'fade_active': False
                    }
                return self._channels[channel_id]
                
        except Exception as e:
            logger.error(f"チャンネル取得エラー: {e}")
            return None

    def fade_volume(self, channel_id: int, target_volume: float, duration: float):
        """音量のフェード処理"""
        try:
            with self._lock:
                if not self._initialized or channel_id not in self._channels:
                    return
                
                channel = self._channels[channel_id]
                state = self._channel_states[channel_id]
                
                # 現在のフェードスレッドを停止
                if channel_id in self._volume_fade_threads:
                    state['fade_active'] = False
                    self._volume_fade_threads[channel_id].join(timeout=0.1)
                
                # 新しいフェードスレッドを開始
                state['target_volume'] = max(0.0, min(1.0, target_volume))
                state['fade_active'] = True
                
                fade_thread = threading.Thread(
                    target=self._fade_volume_thread,
                    args=(channel_id, duration),
                    daemon=True
                )
                self._volume_fade_threads[channel_id] = fade_thread
                fade_thread.start()
                
        except Exception as e:
            logger.error(f"音量フェード処理エラー: {e}")

    def _fade_volume_thread(self, channel_id: int, duration: float):
        """音量フェードスレッド"""
        try:
            channel = self._channels[channel_id]
            state = self._channel_states[channel_id]
            start_volume = state['volume']
            target_volume = state['target_volume']
            start_time = time.time()
            
            while state['fade_active']:
                current_time = time.time()
                elapsed = current_time - start_time
                
                if elapsed >= duration:
                    with self._lock:
                        if state['fade_active']:
                            channel.set_volume(target_volume)
                            state['volume'] = target_volume
                            state['fade_active'] = False
                    break
                
                # 現在の音量を計算
                progress = elapsed / duration
                current_volume = start_volume + (target_volume - start_volume) * progress
                
                with self._lock:
                    if state['fade_active']:
                        channel.set_volume(current_volume)
                        state['volume'] = current_volume
                
                time.sleep(0.01)  # CPU負荷低減
                
        except Exception as e:
            logger.error(f"音量フェードスレッドエラー: {e}")
            with self._lock:
                state = self._channel_states[channel_id]
                state['fade_active'] = False

    def stop_channel(self, channel_id: int):
        """チャンネルの停止"""
        try:
            with self._lock:
                if channel_id in self._channels:
                    channel = self._channels[channel_id]
                    state = self._channel_states[channel_id]
                    
                    # フェードを停止
                    state['fade_active'] = False
                    if channel_id in self._volume_fade_threads:
                        self._volume_fade_threads[channel_id].join(timeout=0.1)
                    
                    # チャンネルを停止
                    channel.stop()
                    state['volume'] = 0.0
                    state['target_volume'] = 0.0
                    
        except Exception as e:
            logger.error(f"チャンネル停止エラー: {e}")

    def cleanup(self):
        """リソースの解放"""
        try:
            with self._lock:
                # すべてのフェードを停止
                for channel_id in self._channel_states:
                    self._channel_states[channel_id]['fade_active'] = False
                
                # フェードスレッドの終了待機
                for thread in self._volume_fade_threads.values():
                    thread.join(timeout=0.2)
                
                # チャンネルの停止
                for channel in self._channels.values():
                    channel.stop()
                
                # リソースのクリア
                self._channels.clear()
                self._sounds.clear()
                self._volume_fade_threads.clear()
                self._channel_states.clear()
                
                # PyGameミキサーの終了
                pygame.mixer.quit()
                self._initialized = False
                
                logger.info("音声システムを終了しました")
                
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
        self.motion_metrics = MotionMetrics(history_size=5)
        self._lock = threading.Lock()
        self.last_hit_time = 0
        self.last_position = None
        self.hit_cooldown = 0.1  # 100ms
        self.last_distance = float('inf')
        self.is_in_contact = False
        self.debug_info = {}
        
        # 音声状態の初期化
        self._initialize_sound_state()
        
    def _initialize_sound_state(self):
        """音声状態の初期化"""
        self.current_volume = 0.0
        self.target_volume = 0.0
        self.fade_duration = 0.1
        self.is_playing = False
        self.is_in_contact = False
        self.last_contact_time = 0
        self.last_detection_time = time.time()
        
        # 音声効果パラメータ
        self.effect_params = {
            'resonance': 1.0,
            'pitch': 1.0,
            'low_pass': 1000,
            'high_pass': 20,
            'reverb': 0.3
        }
        
        # 音声ファイルの読み込み
        self._load_sound_files()
    
    def _load_sound_files(self):
        """音声ファイルの読み込み"""
        sound_files = {
            'close': 'music/A00.mp3',
            'far': 'music/A00.mp3',
            'impact': 'music/A00.mp3'
        }

        for sound_id, file_path in sound_files.items():
            if not self.audio_manager.load_sound(f"{self.hand_type}_{sound_id}", file_path):
                logger.warning(f"{sound_id} 音声の読み込みに失敗しました")

    def _process_interaction(self, distance: float, motion_data: Tuple):
        """接触判定と音声処理"""
        try:
            filtered_position, velocity, acceleration, motion_metrics = motion_data
            
            # 接触判定
            contact_threshold = self.params.get_adjusted_distance_threshold()
            self.is_in_contact = distance < contact_threshold
            
            # デバッグ情報の更新
            self.debug_info = {
                'distance': distance,
                'contact_threshold': contact_threshold,
                'is_in_contact': self.is_in_contact,
                'velocity': np.linalg.norm(velocity) if velocity is not None else 0,
                'acceleration': np.linalg.norm(acceleration) if acceleration is not None else 0
            }
            
            if DEBUG_MODE:
                logger.debug(f"{self.hand_type} Debug: {self.debug_info}")
            
            # ヒット判定
            is_hit, intensity = self._detect_hit(distance, motion_metrics)
            
            if is_hit and intensity is not None:
                # ヒット時の処理
                if time.time() - self.last_contact_time > 0.1:  # 100ms以上間隔
                    logger.debug(f"{self.hand_type} Hit! Intensity: {intensity:.3f}")
                    self._play_impact_sound(intensity)
                    self.last_contact_time = time.time()
            
            # 継続的な音量制御
            if self.is_in_contact:
                target_volume = self._calculate_ambient_volume(distance)
                self.target_volume = target_volume
            else:
                self.target_volume = 0.0
            
            # 音量の更新
            self._update_volume()
            
        except Exception as e:
            logger.error(f"接触処理エラー: {e}")
            self.is_in_contact = False

    def _detect_hit_motion(self, speed: float, acceleration: float, distance: float) -> bool:
        """叩く動作の検出（改善版）"""
        try:
            # 基本的な閾値
            SPEED_THRESHOLD = 0.5  # m/s
            ACCEL_THRESHOLD = 5.0  # m/s²
            DISTANCE_THRESHOLD = 0.1  # m
            
            # 各条件のチェック
            speed_check = speed > SPEED_THRESHOLD
            accel_check = acceleration > ACCEL_THRESHOLD
            distance_check = distance < DISTANCE_THRESHOLD
            
            if DEBUG_MODE:
                logger.debug(f"{self.hand_type} Motion - Speed: {speed:.2f}, Accel: {acceleration:.2f}, Distance: {distance:.2f}")
                logger.debug(f"Checks - Speed: {speed_check}, Accel: {accel_check}, Distance: {distance_check}")
            
            # すべての条件を満たす場合にヒットと判定
            return speed_check and accel_check and distance_check
            
        except Exception as e:
            logger.error(f"ヒット検出エラー: {e}")
            return False

    def update_sound(self, distance: float, current_position: np.ndarray):
        """サウンド状態の更新（改善版）"""
        try:
            current_time = time.time()
            
            # モーション解析
            filtered_position, velocity, acceleration = self.motion_metrics.update(current_position)
            if filtered_position is None:
                return

            # 速度と加速度の計算
            if velocity is not None and acceleration is not None:
                speed = np.linalg.norm(velocity)
                accel = np.linalg.norm(acceleration)
                
                # 急激な動きの検出（叩く動作）
                is_hit = self._detect_hit_motion(speed, accel, distance)
                
                if is_hit and current_time - self.last_hit_time > self.hit_cooldown:
                    # 衝撃音の再生
                    hit_intensity = self._calculate_hit_intensity(speed, accel)
                    self._play_impact_sound(hit_intensity)
                    self.last_hit_time = current_time
                    
                    if DEBUG_MODE:
                        logger.debug(f"{self.hand_type} Hit detected - Speed: {speed:.2f}, Accel: {accel:.2f}, Intensity: {hit_intensity:.2f}")

            self.last_position = filtered_position
            self.last_distance = distance
            
        except Exception as e:
            logger.error(f"サウンド更新エラー: {e}")

    def _validate_input(self, distance: Optional[float], position: Optional[np.ndarray]) -> bool:
        """入力値の検証"""
        if distance is None or position is None:
            return False
        
        if not np.isfinite(distance) or distance < 0:
            return False
        
        if not np.all(np.isfinite(position)):
            return False
        
        return True

    def _analyze_motion(self, position: np.ndarray) -> Optional[Tuple]:
        """モーション解析"""
        try:
            filtered_position, velocity, acceleration = self.motion_metrics.update(position)
            if filtered_position is None:
                return None
            
            motion_metrics = self.motion_metrics.get_average_metrics()
            return filtered_position, velocity, acceleration, motion_metrics
            
        except Exception as e:
            logger.error(f"モーション解析エラー: {e}")
            return None

    def _process_hit_detection(self, distance: float, motion_data: Tuple):
        """ヒット検出と音量制御（改善版）"""
        try:
            _, _, _, motion_metrics = motion_data
            is_hit, intensity = self._detect_hit(distance, motion_metrics)
            
            current_time = time.time()
            if is_hit and intensity is not None:
                time_since_last_contact = current_time - self.last_contact_time
                if time_since_last_contact > 0.5:  # クールダウン時間
                    # 音量の急激な変化を防ぐ
                    target_volume = self._calculate_hit_volume(intensity)
                    smooth_factor = 0.3
                    self.target_volume = (
                        self.target_volume * (1 - smooth_factor) +
                        target_volume * smooth_factor
                    )
                    
                    self.last_contact_time = current_time
                    self._play_impact_sound(min(intensity, 0.8))  # 音量を制限
            else:
                # 環境音の音量をよりスムーズに変更
                ambient_volume = self._calculate_ambient_volume(distance)
                self.target_volume = self.target_volume * 0.9 + ambient_volume * 0.1
            
            # 音量の更新
            self._update_volume()
            
        except Exception as e:
            logger.error(f"ヒット処理エラー: {e}")

    def _detect_hit(self, distance: float, motion_metrics: Tuple[float, float, float]) -> Tuple[bool, Optional[float]]:
        """ヒット検出（デバッグ強化版）"""
        try:
            avg_velocity, avg_acceleration, vertical_alignment = motion_metrics
            
            if DEBUG_MODE:
                logger.debug(f"Hit Detection - {self.hand_type}:")
                logger.debug(f"  Distance: {distance:.3f}m")
                logger.debug(f"  Velocity: {avg_velocity:.3f}")
                logger.debug(f"  Acceleration: {avg_acceleration:.3f}")
                logger.debug(f"  Vertical Alignment: {vertical_alignment:.3f}")

            # 判定条件の確認
            distance_check = distance < self.params.get_adjusted_distance_threshold()
            velocity_check = avg_velocity >= self.params.VELOCITY_THRESHOLD
            acceleration_check = avg_acceleration >= self.params.ACCELERATION_THRESHOLD
            direction_check = vertical_alignment >= self.params.DIRECTION_THRESHOLD

            if DEBUG_MODE:
                logger.debug("Check Results:")
                logger.debug(f"  Distance: {distance_check}")
                logger.debug(f"  Velocity: {velocity_check}")
                logger.debug(f"  Acceleration: {acceleration_check}")
                logger.debug(f"  Direction: {direction_check}")

            is_hit = all([distance_check, velocity_check, acceleration_check, direction_check])

            if is_hit:
                intensity = self._calculate_hit_intensity(avg_velocity, avg_acceleration)
                logger.debug(f"Hit Detected! Intensity: {intensity:.3f}")
                return True, intensity

            return False, None

        except Exception as e:
            logger.error(f"ヒット検出エラー: {e}")
            return False, None

    def _calculate_hit_intensity(self, speed: float, acceleration: float) -> float:
        """ヒット強度の計算（改善版）"""
        try:
            # 速度と加速度から強度を計算
            speed_factor = min(speed / 2.0, 1.0)  # 2 m/s で最大
            accel_factor = min(acceleration / 10.0, 1.0)  # 10 m/s² で最大
            
            # 両方の要素を組み合わせて強度を計算
            intensity = (speed_factor + accel_factor) / 2
            
            # 0.2から1.0の範囲に収める
            return max(0.2, min(1.0, intensity))
            
        except Exception as e:
            logger.error(f"強度計算エラー: {e}")
            return 0.3  # エラー時のデフォルト値

    def _calculate_hit_volume(self, intensity: float) -> float:
        """ヒット音量の計算"""
        try:
            base_volume = intensity * 0.8
            resonance_boost = self.effect_params['resonance'] * 0.2
            return min(1.0, base_volume + resonance_boost)
        except Exception as e:
            logger.error(f"ヒット音量計算エラー: {e}")
            return 0.0

    def _calculate_ambient_volume(self, distance: float) -> float:
        """環境音の音量計算"""
        try:
            if distance > self.params.BASE_DISTANCE_THRESHOLD * 2:
                return 0.0
            
            # 距離に基づく減衰
            attenuation = 1.0 - (distance / (self.params.BASE_DISTANCE_THRESHOLD * 2))
            return float(max(0.0, min(0.3, attenuation * 0.3)))
            
        except Exception as e:
            logger.error(f"環境音量計算エラー: {e}")
            return 0.0

    def _update_volume(self):
        """音量の更新"""
        try:
            current_time = time.time()
            delta_time = current_time - self.last_update_time
            
            if delta_time <= 0:
                return
            
            # 音量変化の計算
            volume_change = (self.target_volume - self.current_volume)
            fade_factor = min(delta_time / self.fade_duration, 1.0)
            self.current_volume = max(0.0, min(1.0, 
                                             self.current_volume + volume_change * fade_factor))
            
            # チャンネルの制御
            channel = self.audio_manager.get_channel(self.channel_id)
            if channel:
                if self.current_volume > 0.001:
                    if not self.is_playing:
                        self._start_sound()
                    channel.set_volume(float(self.current_volume))
                else:
                    if self.is_playing:
                        self._stop_sound()
                        
        except Exception as e:
            logger.error(f"音量更新エラー: {e}")

    def _start_sound(self):
        """音の再生開始"""
        try:
            channel = self.audio_manager.get_channel(self.channel_id)
            if channel:
                sound_id = f"{self.hand_type}_close"
                sound = self.audio_manager._sounds.get(sound_id)
                if sound:
                    channel.play(sound, loops=-1)  # 継続的な再生
                    self.is_playing = True
                    logger.debug(f"音声再生開始: {sound_id}")
        except Exception as e:
            logger.error(f"音声再生開始エラー: {e}")

    def _stop_sound(self):
        """音の停止"""
        try:
            channel = self.audio_manager.get_channel(self.channel_id)
            if channel:
                channel.stop()
                self.is_playing = False
                logger.debug(f"音声停止: {self.hand_type}")
        except Exception as e:
            logger.error(f"音声停止エラー: {e}")

    def _play_impact_sound(self, intensity: float):
        """衝撃音の再生（改善版）"""
        try:
            # 専用チャンネルの取得
            impact_channel = self.audio_manager.get_channel(self.channel_id + 2)
            if not impact_channel:
                return
            
            # サウンドの取得と再生
            sound_id = f"{self.hand_type}_impact"
            sound = self.audio_manager._sounds.get(sound_id)
            if sound:
                # 既存の再生を停止
                impact_channel.stop()
                
                # 音量を設定して再生
                impact_channel.set_volume(intensity)
                impact_channel.play(sound)
                
                if DEBUG_MODE:
                    logger.debug(f"{self.hand_type} Impact sound played - Intensity: {intensity:.2f}")
            else:
                logger.warning(f"Impact sound not found: {sound_id}")
                
        except Exception as e:
            logger.error(f"衝撃音再生エラー: {e}")

    def _handle_no_detection(self):
        """検出なし時の処理"""
        try:
            current_time = time.time()
            if current_time - self.last_detection_time > 0.5:  # 0.5秒のタイムアウト
                self.target_volume = 0.0
                if self.is_playing:
                    self._stop_sound()
                self.motion_metrics = MotionMetrics()  # モーション状態のリセット
                self.is_in_contact = False
                
        except Exception as e:
            logger.error(f"検出なし処理エラー: {e}")

    def _update_effect_parameters(self, distance: float, motion_metrics: Tuple[float, float, float]):
        """音響効果パラメータの更新"""
        try:
            avg_velocity, avg_acceleration, _ = motion_metrics
            
            # 共鳴効果の計算
            self.effect_params['resonance'] = 1.0 - (
                distance / self.params.BASE_DISTANCE_THRESHOLD
            )
            self.effect_params['resonance'] = max(0.0, min(1.0, 
                                                         self.effect_params['resonance']))
            
            # ピッチの計算
            velocity_factor = avg_velocity / self.params.VELOCITY_THRESHOLD
            self.effect_params['pitch'] = 1.0 + (velocity_factor * 0.2)
            
            # フィルター設定の更新
            self.effect_params.update({
                'low_pass': 1000 + (avg_acceleration * 500),
                'reverb': 0.3 + (self.effect_params['resonance'] * 0.2)
            })
            
        except Exception as e:
            logger.error(f"エフェクトパラメータ更新エラー: {e}")

    def check_timeout(self):
        """検出タイムアウトのチェック"""
        try:
            if time.time() - self.last_detection_time > 0.5:
                self._handle_no_detection()
        except Exception as e:
            logger.error(f"タイムアウトチェックエラー: {e}")

    def cleanup(self):
        """リソースの解放"""
        try:
            with self._lock:
                if self.is_playing:
                    self._stop_sound()
                self.motion_metrics = MotionMetrics()
                logger.info(f"SoundController ({self.hand_type}) のリソースを解放しました")
        except Exception as e:
            logger.error(f"SoundController クリーンアップエラー: {e}")