# Experiments

## Running Experiments

1. Basic training:
```bash
python train.py --epochs 100 --batch_size 32
```

2. With configuration file:
```bash
python train.py --config configs/ieee_tac.yaml
```

## Results
- Current implementation: ~20% accuracy (optimization ongoing)
- Target performance: 70%+ accuracy
- Novel metrics: ERCS (Emotion Recognition Confidence Score) and CEDI (Cross-Emotion Discrimination Index)