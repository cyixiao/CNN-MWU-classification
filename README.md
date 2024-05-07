# Optimizing Image Classification Accuracy with MWU on Neural Networks

## Overview

This project explores an ensemble-based approach to improve image classification accuracy using the CIFAR-10 dataset. By leveraging the Multiplicative Weights Update (MWU) algorithm across four convolutional neural networks (CNNs), we aim to boost classification performance through diverse loss functions, weight update methods, and prediction strategies.

## Dataset

The CIFAR-10 dataset comprises 60,000 color images of 32x32 resolution across ten distinct classes. For model training and evaluation, the dataset is divided into five training batches and one test batch.

## Methodology
### Models
Four CNN architectures:
- Standard alternating convolutional and pooling layers.
- VGG-like consecutive convolutional layers followed by a pooling layer.
### Loss Functions
- Symmetrical Reward-Penalty Loss
- Asymmetrical Penalty-Heavy Loss
- Asymmetrical Reward-Heavy Loss
- Weighted Majority Algorithm (WMA) Loss
- Cross-Entropy Loss
- Mean Squared Error (MSE) Loss
### Weight Update Methods
- Traditional MWU Weight Update
- Exponential Weighting
- Additive Weighting
- Adaptive Epsilon
### Prediction Strategies
- Random Selection Based on Normalized Weights
- Weighted Voting Based on Cumulative Weights
- Bayesian Model Averaging

## Results
Our approach achieved an ensemble accuracy of 76%, which is 3% higher than the best individual model. Bayesian Model Averaging showed superior performance across different setups.

## Conclusion and Future Work
The project demonstrates the potential of ensemble learning in improving prediction accuracy in image classification tasks. Future work will focus on refining CNN architectures, exploring additional algorithms, and expanding the application to other datasets.

## Access the Full Report
For a detailed analysis, please refer to our [Report.pdf](./Report.pdf).
