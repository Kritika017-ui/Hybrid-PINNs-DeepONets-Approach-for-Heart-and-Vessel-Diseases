# Hybrid-PINNs-DeepONets-Approach-for-Heart-and-Vessel-Diseases
Cardiovascular Modeling Using PINNs & DeepONet

This repository implements a **hybrid physics-informed and deep learning pipeline** for modeling cardiovascular flows and detecting common cardiovascular diseases: **aneurysm, stenosis, atherosclerosis, and valve defects**. It integrates **Physics-Informed Neural Networks (PINNs)**, **DeepONet surrogates**, and a **classification module**, with a **Gradio-based interactive dashboard**.

---

## Table of Contents

* [Overview](#overview)
* [Pipeline](#pipeline)
* [Installation](#installation)
* [Usage](#usage)
* [File Structure](#file-structure)
* [Evaluation & Experiments](#evaluation--experiments)
* [Milestones](#milestones)
* [References](#references)

---

## Overview

This project provides:

1. **Patient-specific flow modeling** – Predict velocity (`u,v`), pressure (`p`), and wall shear stress (WSS) from a 2D medical slice.
2. **Fast surrogate modeling** – Learn a DeepONet operator to predict flow fields from geometry + boundary conditions in real-time.
3. **Disease classification** – Extract flow-based features and classify diseases.
4. **Interactive interface** – Upload slices, visualize flow fields, and get disease predictions.

---

## Pipeline

### Step 1 — Preprocessing

* **Input:** 2D medical image + mask
* **Output:** Cleaned mask, signed distance function (SDF), interior & boundary points, inlet coordinates, evaluation grid
* **Tools:** Python, numpy, scipy, scikit-image, OpenCV

### Step 2 — PINN Training

* **Goal:** Predict `u(x,y)`, `v(x,y)`, `p(x,y)` using Navier–Stokes equations
* **Output:** Trained PINN checkpoint, field maps (`u.npy`, `v.npy`, `p.npy`), WSS map

### Step 3 — Dataset Creation

* **Goal:** Create datasets for DeepONet
* **Output:** Downsampled field maps (`mask_64.npy`, `u_64.npy`, etc.) and metadata (`index.csv`)

### Step 4 — DeepONet Surrogate

* **Goal:** Map `geometry + BC → u,v,p` quickly
* **Architecture:** Branch net (CNN) + Trunk net (MLP)
* **Output:** Trained DeepONet checkpoint

### Step 5 — Feature Extraction & Classifier

* **Features:** WSS statistics, pressure drop, velocity/vorticity, recirculation fraction, geometry features
* **Classifier:** RandomForest, XGBoost, or CNN
* **Output:** Trained classifier and evaluation metrics

### Step 6 — Gradio Dashboard

* **Goal:** Interactive UI for image upload → segmentation → field prediction → disease classification
* **Output:** Velocity, pressure, WSS visualizations, predicted label & probability, downloadable results

---

## Installation

```bash
# Clone repository
git clone https://github.com/<username>/cardio-pinn-deeponet.git
cd cardio-pinn-deeponet

# Create environment
conda create -n cardio python=3.10
conda activate cardio

# Install dependencies
pip install -r requirements.txt
```

**Recommended `requirements.txt` packages:**

```
numpy
scipy
scikit-image
opencv-python
matplotlib
torch
torchvision
gradio
deepxde
```



## File Structure

```
project/
  data/
    raw/
    preproc/
      case_0001/
        image.png
        mask.png
        sdf.npy
        pts_interior.npy
        pts_boundary.npy
        inlet_mask.npy
  pinn_outputs/
    case_0001/
      u_128.npy
      v_128.npy
      p_128.npy
      wss_128.npy
  deepo_dataset/
    case_0001/
      mask_64.npy
      u_64.npy
      v_64.npy
      p_64.npy
  models/
    pinn_case_0001.pth
    deeponet.pth
    classifier.pkl
  notebooks/
    01_preprocess.ipynb
    02_train_pinn.ipynb
    03_train_deeponet.ipynb
    04_train_classifier.ipynb
    05_dashboard.ipynb
```

---

## Evaluation & Experiments

* **PINN:** Divergence mean, momentum residual, visual inspection
* **DeepONet:** RMSE(u), RMSE(p), divergence, inference time
* **Classifier:** Accuracy, F1, AUC, confusion matrix
* **Ablation studies:** Physics vs data-only, DeepONet trained on PINN vs CFD outputs

---

## Suggested Milestones

1. Preprocess 10 slices → verify masks, SDF, and collocation points
2. Train per-geometry PINN on 1–2 cases → save outputs
3. Build dataset of ~50 PINN outputs (64×64)
4. Train DeepONet → test on held-out cases
5. Extract features → train RandomForest classifier → evaluate

---


