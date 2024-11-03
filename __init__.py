# __init__.py

# 导入主要类和函数
from .device import IoTDevice, CustomDataset
from .network import IoTNetwork
from .experiment import Experiment

__all__ = [
    "IoTDevice",
    "CustomDataset",
    "IoTNetwork",
    "Experiment"
]

# 模块元数据
__version__ = "0.1.0"  # 当前模块版本
__author__ = "您的名字"  # 作者信息
__description__ = "A package for federated learning with IoT devices."  # 模块描述

# 如果需要，还可以添加初始化逻辑，例如设置日志、配置等
import logging

def setup_logging(level=logging.INFO):
    """设置日志配置"""
    logging.basicConfig(level=level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.info("Logging is set up.")

# 运行初始化设置
setup_logging()