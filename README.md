# Hierarchical Emotional Intelligence Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **"Hierarchical Emotional Intelligence through Joint Embedding Predictive Architecture and Active Inference"** submitted to IEEE Transactions on Affective Computing.

## 📋 Abstract

This work presents a novel framework for emotional intelligence that combines Joint Embedding Predictive Architecture (JEPA) with Active Inference principles, treating emotional dynamics as hierarchical free energy minimization processes. We formalize valence as the first-order derivative (dF/dt) and arousal as the second-order derivative (d²F/dt²) of free energy.

## 🚀 Key Features

- **JEPA-Active Inference Integration**: Novel architecture combining self-supervised learning with Bayesian brain principles
- **Hierarchical Processing**: Multi-level emotional understanding with temporal abstraction
- **Novel Metrics**: ERCS (Emotional Response Coherence Score) and CEDI (Contextual Emotional Dynamics Index)
- **State-of-the-art Performance**: 71.3% accuracy on FER2013 with EfficientNet-B0
- **Free Energy Formulation**: Mathematical framework for emotional dynamics

## 📦 Installation
```bash
# Clone the repository
git clone https://github.com/HyungseokSeo/hierarchical-ei.git
cd hierarchical-ei

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
pip install -r requirements.txt
```

## 🎯 Quick Start
```python
from heei.core import HierarchicalEmotionalIntelligence
from heei.metrics import ERCS, CEDI

# Initialize model
model = HierarchicalEmotionalIntelligence(
    hierarchy_levels=3,
    embedding_dim=768,
    use_active_inference=True
)

# Load pretrained weights
model.load_state_dict(torch.load('checkpoints/best_model.pth'))

# Inference
emotions = model.predict(input_tensor)
print(f"Valence: {emotions['valence']}, Arousal: {emotions['arousal']}")
```

## 📊 Reproduce Paper Results
```bash
# Download datasets
python scripts/download_data.py

# Train the full model
python experiments/train_hierarchical.py \
    --config experiments/configs/ieee_tac.yaml \
    --dataset fer2013 \
    --epochs 100

# Evaluate with our metrics
python experiments/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --metrics ercs cedi accuracy
```

## 📈 Results

| Model | Dataset | Accuracy | ERCS | CEDI |
|-------|---------|----------|------|------|
| HEEI (Ours) | FER2013 | **71.3%** | **0.842** | **0.789** |
| EfficientNet-B0 | FER2013 | 70.1% | 0.751 | 0.698 |
| ResNet-50 | FER2013 | 68.5% | 0.723 | 0.672 |

## 🧬 Model Architecture
```
Input → JEPA Encoder → Hierarchical Processing → Active Inference Module
                              ↓
                     Free Energy Computation
                              ↓
                    Emotional Dynamics (dF/dt, d²F/dt²)
                              ↓
                      Valence & Arousal Output
```

## 📚 Citation

If you find this work useful, please cite:
```bibtex
@article{seo2025hierarchical,
  title={Hierarchical Emotional Intelligence through Joint Embedding Predictive Architecture and Active Inference},
  author={Seo, Hyungseok},
  journal={IEEE Transactions on Affective Computing (submitted)},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Contact

- **Author**: Hyungseok Seo
- **Email**: henry_seo@naver.com
- **Lab**: Chungbuk National University, Rep. of Korea
- **Advisor**: Prof. Sung-Jin Kim

## 🙏 Acknowledgments

The author thanks Professor Sung-Jin Kim for invaluable guidance. Computational resources were provided by Chungbuk National University’s AI Research Center.
I also thank the reviewers for their valuable feedback and the open-source community for the tools that made this research possible.
