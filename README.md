
# Handwritten Digit Video Localization & Prediction 🎥🔢

[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20TensorFlow-orange.svg)]()
[![Model](https://img.shields.io/badge/Model-cGAN%20%2B%20ConvGRU-blue.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-Custom%20Moving%20MNIST-lightgrey.svg)]()

This repository contains the implementation of an advanced spatiotemporal video prediction model designed to forecast complex, multi-digit spatial bounding-box trajectories. The model leverages a hybrid conditional Generative Adversarial Network (cGAN) architecture augmented with recurrent and attention mechanisms to predict future frames based on historical context.

## 🚀 Key Features

* **Custom Moving MNIST Dataset:** Curated a specialized dataset comprising 2,000 training and 100 testing sequences.
* **Context-Aware Forecasting:** Utilizes a 15-frame input context to accurately generate a 5-frame future target.
* **Advanced Architecture:** Integrates a U-Net-inspired conditional GAN with custom Convolutional Gated Recurrent Units (ConvGRU).
* **Spatiotemporal Attention:** Employs an 8-head Spatiotemporal Attention mechanism to capture complex temporal dynamics and spatial transformations.
* **High Robustness:** Rigorously tested against environmental anomalies and visual distortions.

## 📊 Dataset Specifications

The model is trained and evaluated on a custom-built version of the Moving MNIST dataset, designed specifically for multi-digit bounding-box trajectory modeling.

* **Training Set:** 2,000 sequences
* **Testing Set:** 100 sequences
* **Input Window:** 15 consecutive frames
* **Prediction Horizon:** 5 future frames
* **Complexity:** Sequences contain 1, 2, and 3 simultaneously moving digits.

## 🧠 Model Architecture

The core of the prediction engine is a hybrid **U-Net-inspired conditional GAN (cGAN)**. 

To effectively capture spatial hierarchies and temporal dependencies across frames, the generator incorporates:
1. **Custom ConvGRU Layers:** To retain temporal memory and motion trajectories across the 15-frame input sequence.
2. **8-Head Spatiotemporal Attention:** To allow the model to focus on critical moving objects and complex crossing trajectories across both space (pixels) and time (frames).

## 📈 Results & Evaluation

The model was benchmarked against standard baseline predictive models for sequences containing 1, 2, and 3 moving digits. 

### Quantitative Metrics
* **Average Structural Similarity Index (SSIM):** `0.7235`
* **Average Peak Signal-to-Noise Ratio (PSNR):** `17.51 dB`

### Robustness & Anomaly Testing
The model demonstrates sustained sequence prediction accuracy and resilience when subjected to complex, simulated environmental anomalies, including:
* 🌑 **Low Light Conditions**
* 💨 **Motion Blur**
* 📺 **Background Noise**
* 📐 **Perspective Distortions**

## 🛠️ Installation & Usage

*(Note: Add your specific environment setup steps here)*

```bash
# Clone the repository
git clone [https://github.com/yourusername/moving-mnist-prediction.git](https://github.com/yourusername/moving-mnist-prediction.git)
cd moving-mnist-prediction

# Install dependencies
pip install -r requirements.txt

# Run inference on test set
python evaluate.py --model_weights path/to/weights.pth --dataset_path data/test_set/
