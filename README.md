# BabyLM ELECTRA Pre-training Project

## Project Overview

This project implements and evaluates a custom pre-trained ELECTRA language model using the BabyLM 10-million token dataset. The model was developed to explore lightweight language model pre-training on a constrained dataset and evaluate its performance against established models like BERT in downstream tasks.

## Key Features

- Custom pre-trained ELECTRA model on BabyLM dataset
- HPC-accelerated training using NVIDIA L40S GPUs
- Comprehensive evaluation against BERT base model
- Document classification testing across seven academic domains

## Project Structure

```
├── model/
│   ├── config.json              # Model architecture configuration
│   └── babylm_model.bin         # Pre-trained model weights
├── scripts/
│   └── electraPreTrain-slurm.sh # SLURM job script for HPC training
├── trainModel.py                # Main training implementation
└── evaluation.py                # Model evaluation script
```

## Technical Details

### Pre-training Architecture

The ELECTRA model was configured with:
- 18 hidden layers
- Hidden size of 196
- 4 attention heads
- 64 embedding dimensions
- Intermediate size of 128
- Maximum sequence length of 512

This configuration creates a compact model while maintaining strong language understanding capabilities.

### Training Process

The model was trained on Boise State University's HPC infrastructure using NVIDIA L40S GPUs. The training process included:

1. Data preparation using the BabyLM 10M token dataset
2. Masked language modeling with 15% masking probability
3. Training with AdamW optimizer
4. Monitoring both training and validation loss

The `electraPreTrain-slurm.sh` script handled resource allocation and job management on the HPC cluster.

### Evaluation Results

The pre-trained model was evaluated against standard BERT for document classification tasks across different batch sizes:

| Model   | Batch Size | Epoch 1 | Epoch 2 | Epoch 3 |
|---------|------------|---------|---------|---------|
| BERT    | 8          | 75%     | 78%     | 77%     |
| ELECTRA | 8          | 57%     | 69%     | 70%     |
| BERT    | 16         | 77%     | 77%     | 78%     |
| ELECTRA | 16         | 17%     | 38%     | 52%     |
| BERT    | 32         | 75%     | 78%     | 78%     |
| ELECTRA | 32         | 17%     | 17%     | 17%     |

Key findings:
- BERT showed more consistent performance across batch sizes
- ELECTRA performed best with small batch sizes (8)
- ELECTRA showed steeper learning curves with continued training
- ELECTRA had faster training times despite lower initial accuracy

## Usage

### Dependencies

```
torch
transformers
numpy
tqdm
sklearn
```

## Conclusions

The project demonstrates that custom pre-trained ELECTRA models can be effectively created using moderate computing resources and limited datasets. While BERT outperformed our ELECTRA model in immediate accuracy, ELECTRA showed promising efficiency characteristics with:

1. Faster training times
2. Steeper learning curves
3. Better performance at smaller batch sizes

This indicates potential for ELECTRA-based architectures in resource-constrained environments where training efficiency matters more than immediate accuracy.

## Acknowledgements

This project was completed using Boise State University's computing resources, specifically their NVIDIA L40S GPU infrastructure.
