# RUL-Prediction-for-Aero-Engine-Markov-Chain-Hand-coded-Gaussian-HMM-and-XGBoost
基于C-MAPSS航空发动机退化数据集的剩余寿命预测对比实验

## Project Overview
Remaining Useful Life (RUL) prediction is a core task of Predictive Maintenance (PdM).
This project builds three modeling paradigms for degradation modelling, compares their performance, assumptions and interpretability.
The core focus is to explore the trade-off between prior knowledge, model assumption and prediction accuracy.

Three models in contrast:
1. Discrete absorbing Markov chain: Construct degradation states manually by RUL threshold, no sensor observation used.
2. Hand-implemented Gaussian Hidden Markov Model (HMM):
   - Hidden degradation states; sensor signals as Gaussian observations
   - Baum-Welch (EM algorithm) for parameter estimation
   - Viterbi algorithm for state decoding
   - Support absorbing state constraint for failure
3. XGBoost supervised regression baseline: Sliding window statistical features, direct RUL regression.

> Dataset: NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset (FD001)

## Key Metrics
| Model | RMSE | MAE | Notes |
|---|---|---|---|
| XGBoost Baseline | 18.80 | 13.96 | Discriminative regression, no Markov assumption |
| Hand-coded Gaussian HMM | 58.77 | 47.87 | Generative, sensor-driven hidden states |
| Discrete Markov Chain | 61.38 | 51.54 | No sensor data, state built only from RUL label |

## Core Findings
1. Adding sensor observations brings consistent improvement: Gaussian HMM outperforms discrete Markov chain, proving sensor time series contain degradation information.
2. First-order Markov assumption acts as a strong constraint: HMM assumes future state only depends on current hidden state, which limits its fitting capacity compared with discriminative XGBoost.
3. EM algorithm for HMM is sensitive to initialization:
   - Uniform initialization: Better prediction accuracy, but transition matrix is close to uniform, physically uninterpretable (random state jump).
   - Diagonal-dominant prior initialization (degradation monotonicity): Transition matrix matches engineering intuition (stay in current state with high probability, only small chance to degrade further), at the cost of higher prediction error.
4. Discriminative model like XGBoost has no Markov constraint and can learn complex non-linear patterns from window features, hence achieves best accuracy.

## Visualization
1. Transition matrix heatmap comparison
    - Discrete Markov chain: state defined by RUL threshold
    - Gaussian HMM: state learned automatically from sensor data
2. RUL prediction curve on single validation engine: True RUL vs three model outputs, showing the gap between models.

## Technical Stack
Python | NumPy | Matplotlib | Seaborn | Scikit-learn | XGBoost
Core implementation: Hand-written Baum-Welch EM, Viterbi decoding, log-domain calculation to avoid numerical underflow.

## Project Structure
├── test.py # Main script: data preprocessing, 3 models, evaluation, visualization
├── CMAPSSData/ # NASA C-MAPSS dataset
├── figures/ # Output heatmap & prediction curve
└── README.md

## Interview / Discussion Points
- Why EM (Baum-Welch) is prone to local optimum and sensitive to initialization
- Difference between generative HMM and discriminative regression
- The trade-off between model interpretability and predictive performance
- Limitation of first-order Markov assumption on degradation modelling
- Numerical stability trick: log-domain probability calculation to avoid underflow



