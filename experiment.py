import numpy as np
from datasets import load_dataset
from .device import IoTDevice, CustomDataset
from .network import IoTNetwork


class Experiment:
    def __init__(self, model_name: str, num_clients: int, data_path: str, output_dir: str):
        self.model_name = model_name  # 使用的模型名称
        self.num_clients = num_clients  # 总设备数量
        self.data_path = data_path  # 数据路径
        self.output_dir = output_dir  # 输出目录
        self.devices = []  # 存储设备的列表
        self.network = None  # IoT 网络

    def setup_data(self):
        """加载数据并为每个设备分配数据集"""
        dataset = load_dataset("databricks/dolly-15k")  # 加载示例数据集
        data_segments = np.array_split(dataset, self.num_clients)  # 根据设备数量拆分数据集

        for idx, segment in enumerate(data_segments):
            local_data = CustomDataset(segment)  # 将数据切片封装成自定义数据集
            device_type = "low_power" if idx % 2 == 0 else "edge_node"
            learning_rate = 1e-5 if device_type == "low_power" else 1e-4
            device = IoTDevice(device_id=idx + 1, device_type=device_type, learning_rate=learning_rate,
                               model_name=self.model_name, local_data=local_data)
            self.devices.append(device)  # 添加设备到设备列表

    def run_experiment(self, num_communication_rounds: int):
        """执行联邦学习实验"""
        self.network = IoTNetwork(self.devices)  # 初始化 IoT 网络
        self.network.initialize_global_model(self.model_name)  # 初始化全局模型

        for round_num in range(num_communication_rounds):
            print(f"\nCommunication Round {round_num + 1}/{num_communication_rounds}")

            for device in self.devices:
                print(f"Training Device {device.device_id}")
                device.train()  # 在每个设备上进行本地训练

            print("Aggregating updates from devices...")
            self.network.aggregate_updates()  # 聚合本地更新

            print("Distributing global model to devices...")
            self.network.distribute_global_model()  # 分发更新后的全局模型

            print("Evaluating global model...")
            self.network.evaluate_global_model(self.dummy_evaluation_function)  # 评估全局模型

    @staticmethod
    def dummy_evaluation_function(model):
        """伪造的评估函数示例"""
        return {'accuracy': 0.85, 'loss': 0.25}  # 返回固定的评估指标，仅用于示例


# 如果需要，您可以根据需要添加其他辅助方法或扩展功能

# 可通过设置简单的 CLI 接口进而测试或运行此实验
if __name__ == "__main__":
    experiment = Experiment(model_name='Llama/7B', num_clients=10, data_path='./data', output_dir='./output')
    experiment.setup_data()  # 设置数据
    experiment.run_experiment(num_communication_rounds=5)  # 运行实验