# 项目全局配置

from pathlib import Path

# ============== 路径配置 ==============
BASE_DIR = Path('data')
BUILDINGS_DIR = BASE_DIR / 'buildings'  # 建筑数据目录
SCALER_DIR = BASE_DIR / 'scalers'
MODEL_DIR = Path('models')

# ============== 建筑配置 ==============
# 建筑列表（在此处统一配置，所有 notebook 共享）
BUILDINGS = ['南阶', '西阶']

# 源域和目标域（根据 similarity_analysis.ipynb 相似性分析结果设置）
SOURCE_BUILDING = '南阶'
TARGET_BUILDING = '西阶'

# 目标域小样本时间段（模拟"小样本"场景）
TARGET_SAMPLE_START = '2024-03-01'
TARGET_SAMPLE_END = '2024-07-01'

# ============== 数据列配置 ==============
TARGET_COL = 'hourly_kwh_clean'  # 目标列（负荷）
TIME_COL = 'timestamp'  # 时间戳列

SEED = 42