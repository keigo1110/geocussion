"""インタラクティブ砂場音響システム
パッケージ初期化
"""

from .core import (
    SystemParameters,
    ResourceManager,
    PerformanceMonitor,
    initialize_system,
    DEBUG_MODE,
    resource_manager
)

from .camera import (
    RealSenseManager,
    FrameProcessor
)

from .hand_tracking import (
    MotionMetrics,
    HandTracker
)

from .audio import (
    AudioManager,
    SoundController
)

from .pointcloud import (
    PointCloudProcessor
)

from .main import (
    SandboxSystem,
    SystemMonitor,
    main,
    run_with_error_handling
)

__version__ = '2.0.0'