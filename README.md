# CNN-Based Stock Market Prediction  
### An Empirical Reproducibility Study

This repository presents a **rigorous empirical reproducibility study** of the paper:

> *“S&P 500 Stock’s Movement Prediction using CNN”*  
> Rahul Gupta, Stanford University

The original study reports prediction accuracies as high as **91%** for certain
stocks (e.g. JPM, T+30 horizon).  
The goal of this project is to **reproduce, evaluate, and critically assess**
those results under a **realistic and leakage-free experimental protocol**.

---

## 🎯 Objectives

- Reimplement the **full experimental pipeline** described in the original paper
- Use **publicly available market data** instead of proprietary sources
- Enforce **strict temporal validation** to prevent look-ahead bias
- Compare deep learning models against **explicit naive baselines**
- Determine whether the reported performance can be reproduced **ethically**

---

## 📊 Dataset

- **Source**: Yahoo Finance (via `yfinance`)
- **Ticker**: JPM (JP Morgan)
- **Period**: 2005–2025
- **Frequency**: Daily

### Features (10 channels)
- Raw OHLCV: Open, High, Low, Close, Volume  
- Adjusted OHLCV: Adjusted Open, High, Low, Close, Volume  

No technical indicators or engineered features are used.

---

## 🧱 Data Preparation

- **Sliding window** approach (data augmentation)
- Window size: **256 trading days**
- Forecasting horizon: **T+30**
- Binary labels:
  - `BUY (1)` if future return > 0
  - `SELL (0)` otherwise
- Resulting dataset:
  - **4,747 samples**
  - Input shape: `(256, 10)`
  - Class distribution: ~**62.7% BUY**

A **strict temporal train/test split** is applied:
- Train: windows ending before 2018
- Test: windows ending from 2018 onward

---

## 📉 Baselines

Due to the strong upward drift of equity markets, a naive strategy that always
predicts an upward movement already achieves:

- **Always-BUY accuracy ≈ 62.7%**

This baseline is explicitly reported and used as a reference for all models.
Any model failing to significantly outperform this baseline is considered
non-informative.

---

## 🧠 Models Implemented

### 1. CNN Baseline
- 2 × Conv1D layers
- Global average pooling
- Fully connected classifier

Purpose: deep learning sanity check.

### 2. Paper-like CNN (Gupta)
- 8 × Conv1D layers
- ReLU (first layer), LeakyReLU + BatchNorm (subsequent layers)
- Dropout (keep_prob = 0.6)
- Fully connected layers
- Softmax + cross-entropy loss

### 3. Final Controlled Variant
To push the reproduction as far as possible, a final experiment also includes:
- The **non-standard normalization formula** described in the original paper
- A **2048-dimensional fully connected layer** (via adaptive pooling)

All experiments share the **same training loop, hyperparameters, and evaluation
protocol**.

---

## 🧪 Results Summary

| Model                         | Train Accuracy | Test Accuracy |
|------------------------------|----------------|---------------|
| Naive Always-BUY             | —              | **62.7%**     |
| CNN Baseline                 | ~67%           | **62.7%**     |
| Paper-like CNN               | ~78%           | **62.7%**     |
| Final Controlled CNN         | ~83%           | **62.7%**     |

### Key Observation
All CNN models **learn and overfit the training data**, but **none generalize**
to unseen future data beyond the naive baseline.

---

## 🧠 Interpretation

- Increasing model depth and capacity does **not** improve out-of-sample
  performance under realistic validation.
- The dominant signal captured by the models corresponds to **class imbalance**
  and long-term market drift, not predictive market structure.
- The high accuracies reported in the original paper **cannot be reproduced**
  without introducing methodological artifacts.

The most likely explanation for the reported 91% accuracy is **look-ahead bias**,
potentially caused by shuffling time-series data prior to splitting.

---

## 🧪 Reproducibility & Ethics

This project deliberately enforces:
- No global shuffling across time
- No leakage in normalization
- No hidden preprocessing steps
- Explicit comparison to naive baselines

Negative results obtained under a rigorous protocol are considered **valuable
scientific outcomes**.

---

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install yfinance numpy pandas matplotlib torch scikit-learn
