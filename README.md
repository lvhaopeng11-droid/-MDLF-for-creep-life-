# Trustworthy Multimodal Attention Framework (MDLF)

This repository provides the official PyTorch implementation, the complete dataset, explicit data splits, and isolated reproduction scripts for the paper:

> **"Trustworthy multimodal attention framework under data-scarce regimes: demonstrated on superalloy behavior prediction"**  
> *(Currently under review at npj Computational Materials)*.

---

## 📝 Overview & Core Contributions

Predicting the macroscopic performance (e.g., creep rupture life) of superalloys is notoriously challenging under data-scarce conditions. This repository introduces a **Multimodal Deep Learning Framework (MDLF)** that synergistically fuses high-dimensional microstructural micrographs (SEM) with low-dimensional tabular data (composition and heat treatment parameters) via a cross-modal attention mechanism. 

### 🛡️ Addressing Overfitting in Data-Scarce Regimes
To strictly prevent the deep neural network from overfitting on limited experimental data (~90 physical training samples), our framework is heavily regularized through three fundamental design strategies:
1. **Transfer Learning (Layer Freezing):** Utilizing an ImageNet pre-trained ResNet-18 backbone and explicitly freezing the shallow layers (Stages 1 & 2) to drastically reduce trainable parameters.
2. **Image Tiling (Data Augmentation):** Systematically cropping high-resolution micrographs into smaller tiles, thereby expanding the effective training instances from ~90 to >1,100, forcing the network to learn translation-invariant local features.
3. **Cross-Modal Attention (Physical Constraint):** Utilizing tabular thermodynamic parameters as queries to filter out irrelevant visual background noise, allowing the model to focus strictly on physically meaningful features (e.g., $\delta$-phase morphology).
## 📂 Repository Structure
To guarantee strict replicability and absolute transparency, the codebase is highly modularized into 7 distinct pipelines. Independent researchers can seamlessly verify our benchmark results using the explicit stratified data splits provided.

### `1.Data Processing/`
Contains scripts and logic for initial data cleaning and, most importantly, the stratified data splitting strategy.
*   `Data Stratified Strategy.ipynb`: The exact logic used to partition the dataset, ensuring the distribution of alloy compositions remains balanced across sets.

### `2.TableData/`
Dedicated to the explicitly split datasets and traditional machine learning baselines (using Tabular Data Only).
*   **Explicit Data Splits:** `split_train_data.csv`, `split_val_data.csv`, `split_test_data.csv` (These fixed indices guarantee zero data leakage or random splitting bias during reproduction).
*   `Run_tabular_baselines.ipynb`: Script to reproduce traditional ML benchmarks (e.g., XGBoost, SVR, Random Forest).
*   `Run_tabpfn_baseline.py.ipynb`: Script to reproduce the state-of-the-art **TabPFN** foundation model benchmark.

### `3.ImageData/`
Dedicated to image preprocessing and Unimodal Vision baselines.
*   `Split_Images/` & `GHdata/`: Directories handling the cropped microstructural tiles.
*   `Run_image_only_cnn.ipynb`: Script to duplicate the results of the Unimodal CNN (ResNet-18) using only SEM images.

### `4.MDLF/`
Contains the core implementation of our proposed Multimodal Deep Learning Framework.
*   `Run_multimodal_mdlf.py`: The standalone script to train, validate, and test the MDLF model using the cross-modal attention mechanism.
*   `Transfer learning_train.ipynb`: Demonstrates the implementation of our strategic layer-freezing technique.

### `5.Visualization/`
Focuses on decoding the "black-box" nature of the image feature extraction.
*   `GradCAM_model.ipynb`: Generates Gradient-weighted Class Activation Mapping (Grad-CAM) heatmaps, proving that the model focuses on thermodynamically relevant features (e.g., $\delta$-phase) rather than background noise.

### `6.Interpretable/`
Focuses on the interpretability of tabular features.
*   `SHAP.ipynb`: Conducts Shapley Additive Explanations (SHAP) analysis to quantify the non-linear influence of specific microstructural sizes and volume fractions on creep rupture life.

### `7.Results Image/`
Automated plotting scripts to ensure transparency in how the final manuscript figures were generated.
*   `Bar Chart Results.ipynb`, `Feature visualization.ipynb`, etc.: Scripts to plot the exact evaluation metrics ($R^2$, RMSE, MAPE) and comparison charts presented in the paper.


```text
MDLF-Superalloy-Creep/
│
├── 1.Data Processing/                      # Data Cleaning & Stratification
│   ├── Data Processing.ipynb               # Initial data cleaning and merging
│   └── Data Stratified Strategy.ipynb      # Logic for explicitly stratifying the dataset
│
├── 2.TableData/                            # Tabular Baselines & Explicit Data Splits
│   ├── split_train_data.csv                # Fixed Training set indices (Crucial for reproducibility)
│   ├── split_val_data.csv                  # Fixed Validation set indices
│   ├── split_test_data.csv                 # Fixed Test set indices
│   ├── Run_tabular_baselines.ipynb         # Traditional ML benchmarks (XGBoost, RF, SVR, etc.)
│   ├── Run_tabpfn_baseline.ipynb           # TabPFN foundation model benchmark
│   └── Run_CNN_baseline.ipynb              # Baseline evaluations setup
│
├── 3.ImageData/                            # Unimodal Vision Baselines
│   ├── GHdata/                             # Raw/Processed image directories
│   ├── Split_Images/                       # Tiled sub-images (Data Augmentation)
│   └── Run_image_only_cnn.ipynb            # Unimodal ResNet-18 trained ONLY on images
│
├── 4.MDLF/                                 # Core Proposed Framework
│   ├── Run_multimodal_mdlf.py              # Main script: Trains & evaluates the MDLF model
│   ├── Transfer_learning_train.ipynb       # Implements the strategic layer-freezing technique
│   └── Model_Iteration_Metrics.csv         # Saved loss/metric logs during training
│
├── 5.Visualization/                        # Attention Mechanism Interpretability
│   ├── GradCAM_model.ipynb                 # Generates Grad-CAM heatmaps for the δ-phase
│   ├── Visualization_mechanisms.ipynb      # Maps attention weights to physical phenomena
│   ├── ALL_GradCAM_visualizations/         # Output folder for generated heatmaps
│   └── Attention_Batch_Perfect/            # Selected highly-attentive patch examples
│
├── 6.Interpretable/                        # Tabular Feature Interpretability
│   ├── SHAP.ipynb                          # SHAP value extraction for quantitative feature impact
│   └── data.xlsx                           # Feature summary for SHAP analysis
│
└── 7.Results Image/                        # Automated Plotting Scripts (Transparency in Figures)
    ├── Bar Chart Results.ipynb             # Generates R2, RMSE, MAPE comparison bar charts
    ├── Feature visualization.ipynb         # Plots feature distributions (KDE plots)
    └── Results_of_Attention.ipynb          # Visualizes attention score trends