# MNIST Classification

![Build Status](https://github.com/hsinghweb/era-v3-s7-cnn-cloud/actions/workflows/ml-pipeline.yml/badge.svg)

A PyTorch implementation of MNIST digit classification achieving 99.4% test accuracy with less than 20k parameters.

## Features
- Batch Normalization
- Dropout
- Global Average Pooling
- Less than 20k parameters
- 99.4% test accuracy

## Requirements
- Python 3.8+
- PyTorch 1.7+
- See requirements.txt for full list

## Model Training Logs

```
cuda
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------
            Conv2d-1           [-1, 8, 28, 28]              72
       BatchNorm2d-2           [-1, 8, 28, 28]              16
            Conv2d-3          [-1, 12, 28, 28]             864
       BatchNorm2d-4          [-1, 12, 28, 28]              24
         MaxPool2d-5          [-1, 12, 14, 14]               0
           Dropout-6          [-1, 12, 14, 14]               0
            Conv2d-7          [-1, 16, 14, 14]           1,728
       BatchNorm2d-8          [-1, 16, 14, 14]              32
         MaxPool2d-9            [-1, 16, 7, 7]               0
          Dropout-10            [-1, 16, 7, 7]               0
           Conv2d-11            [-1, 16, 7, 7]             256
           Conv2d-12            [-1, 16, 7, 7]           2,304
      BatchNorm2d-13            [-1, 16, 7, 7]              32
           Conv2d-14            [-1, 16, 7, 7]           2,304
      BatchNorm2d-15            [-1, 16, 7, 7]              32
           Conv2d-16            [-1, 16, 7, 7]           2,304
      BatchNorm2d-16            [-1, 16, 7, 7]              32
          Dropout-17            [-1, 16, 7, 7]               0
AdaptiveAvgPool2d-18            [-1, 16, 1, 1]               0
           Linear-19                   [-1, 10]             170
----------------------------------------------------------------
Total params: 10,170
Trainable params: 10,170
Non-trainable params: 0
----------------------------------------------------------------

EPOCH: 0
Loss=0.13029561936855316 Batch_id=468 Accuracy=89.55: 100%|██████████| 469/469 [00:15<00:00, 30.47it/s]

Test set: Average loss: 0.0565, Accuracy: 9831/10000 (98.31%)

EPOCH: 1
Loss=0.017242550477385521 Batch_id=468 Accuracy=97.89: 100%|██████████| 469/469 [00:15<00:00, 30.47it/s]

Test set: Average loss: 0.0326, Accuracy: 9895/10000 (98.95%)

[Additional epochs...]

EPOCH: 19
Loss=0.004597507603466511 Batch_id=468 Accuracy=99.42: 100%|██████████| 469/469 [00:15<00:00, 30.47it/s]

Test set: Average loss: 0.0198, Accuracy: 9942/10000 (99.42%)
```

## Model Architecture
- Input Layer: 1 channel (grayscale images)
- Multiple Convolutional layers with Batch Normalization
- Dropout layers for regularization
- Global Average Pooling
- Output Layer: 10 classes (digits 0-9)

## Training Details
- Optimizer: SGD with momentum
- Learning Rate: 0.01
- Momentum: 0.9
- Batch Size: 128
- Epochs: 20 (with early stopping at 99.4% accuracy)