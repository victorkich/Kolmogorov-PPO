# Kolmogorov-Arnold Networks (KAN) for Online Reinforcement Learning

## Overview
Kolmogorov-Arnold Networks (KAN) are designed to provide an efficient function approximation method tailored for online reinforcement learning tasks. Inspired by the Kolmogorov-Arnold representation theorem, KANs enable quick adaptation and learning from limited data by decomposing complex functions into simpler, more manageable components. This approach is particularly advantageous in dynamic environments where traditional reinforcement learning methods struggle with computational overhead.

## Key Features
- **Efficient Function Approximation**: KANs leverage the Kolmogorov-Arnold representation to approximate complex functions with minimal computational resources, making them ideal for online learning scenarios.
- **Scalability**: The modular structure of KANs allows for scaling to more complex tasks and environments without significant performance degradation.
- **Flexibility**: KANs can be easily integrated with existing reinforcement learning algorithms, enhancing their performance in environments requiring rapid adaptation.

## Setup
1. Clone the KAN repository:
   ```bash
   git clone https://github.com/yourusername/KAN.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Kolmogorov-PPO
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   
## Usage

### Training a KAN Model

1. Configure your training parameters in the config.yaml file.
2. Start the training process by running:
   ```bash
   python train.py --config config.yaml
   ```
3. Monitor the training progress through the provided logs or TensorBoard.

### Custom Environments
KANs can be adapted to custom environments. Ensure that your environment follows the standard Gym API. Modify the environment parameters in the config.yaml file as needed.

### Contact
For questions, issues, or further discussion, please reach out to:

- Victor A. Kich: victorkich98@gmail.com

### Acknowledgments

This work was developed as part of ongoing research in online reinforcement learning. We thank all contributors and collaborators who have supported this project.

### Citation
If you use this code in your research, please cite the following paper:

```bibtex
@article{kich2024kolmogorov,
  title={Kolmogorov-Arnold Network for Online Reinforcement Learning},
  author={Kich, Victor Augusto and Bottega, Jair Augusto and Steinmetz, Raul and Grando, Ricardo Bedin and Yorozu, Ayano and Ohya, Akihisa},
  journal={arXiv preprint arXiv:2408.04841},
  year={2024}
}
```
