```markdown
# Enhancing IoT Intelligence: A Hierarchical Federated Collaborative Computing Framework for Integrating Generative AI

This project implements a hierarchical federated collaborative computing framework designed to integrate Generative AI within IoT environments. The framework is built on the principles outlined in the paper "Enhancing IoT Intelligence: A Hierarchical Federated Collaborative Computing Framework for Integrating Generative AI".

## Authors

- Chengzhuo Han, School of Cyber Science and Engineering, Southeast University, China
- Tingting Yang, Peng Cheng Laboratory, China
- Zhengqi Cui, Navigation College, Dalian Maritime University, China
- Xin Sun, Navigation College, Dalian Maritime University, China

Contact: 
- {hcz\_dmu, yangtingting820523, czq1006, Sunny\_xin1996}@163.com

## Overview

This codebase provides an implementation of a federated learning framework that enables multiple IoT devices to collaboratively train machine learning models while keeping their data localized. The key components include device management, network communication, and federated training processes.

### Features

- **IoT Device Management**: Simulates individual IoT devices that can train models locally.
- **Federated Learning Network**: Facilitates communication between devices and manages the global model.
- **Experiment Framework**: Orchestrates the setup and execution of federated learning experiments.
- **Easy Model Evaluation**: Provides functionality to evaluate the global model performance after training.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/iot_federated_learning.git
   cd iot_federated_learning
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the necessary environment configured for running models (e.g., PyTorch, Hugging Face Transformers).

## Usage

To run the federated learning experiment, you can execute the following Python script:

```python
from experiment import Experiment

# Initialize the experiment with required parameters
experiment = Experiment(model_name='Llama/7B', num_clients=10, data_path='./data', output_dir='./output')

# Set up the local datasets for each device
experiment.setup_data()

# Begin the federated learning experiment
experiment.run_experiment(num_communication_rounds=5)
```

## Example

You can modify parameters such as `model_name`, `num_clients`, `data_path`, and `output_dir` as per your requirements to customize the experiment setup.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bugs you encounter.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Citation

If you find this work helpful, please cite our paper:

```
@article{han2023enhancing,
  title={Enhancing IoT Intelligence: A Hierarchical Federated Collaborative Computing Framework for Integrating Generative AI},
  author={Chengzhuo Han and Tingting Yang and Zhengqi Cui and Xin Sun},
  journal={IEEE Transactions},
  year={2023}
}
```

For further inquiries, please reach out to the authors via their provided contact information.

```

