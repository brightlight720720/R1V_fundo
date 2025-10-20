# 🧠 R1V_fundo: Fine-Tuned Vision-Language Model for Fundus Image Reasoning

## 📘 Overview

**R1V_fundo** is a customized and fine-tuned implementation of GRPO for VLM open-source framework, adapted for **fundus image analysis and reasoning tasks**.  
This repository integrates medical image understanding and multimodal reasoning capabilities, focusing on diabetic retinopathy grading and fundus disease classification.

---

## 🚀 Key Features

- 🔍 **Fundus Image Reasoning** — Generates visual reasoning steps and diagnostic justifications.  
- 🧩 **Multimodal Input Support** — Processes both images and textual clinical context.  
- 🧠 **GRPO-based Reinforcement Learning** — Uses *Group Relative Policy Optimization* (GRPO) for reward-aligned fine-tuning.  
- ⚙️ **SFT + RL Pipeline Integration** — Supports hybrid supervised and reinforcement learning setups.  
- 📈 **Evaluation Metrics** — Includes AUC, QWK, sensitivity, specificity  

---

## 📂 Repository Structure

```
R1-V/src/r1-v
│
├── configs/               # Model, training, and evaluation configs
├── src/                   # Core model training and evaluation code
├── local_scripts/               # Automation and batch training scripts

```

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/R1V_fundo.git
cd R1V_fundo
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Datasets
Download and place your fundus image datasets under the `data/` folder.  
The structure should look like:
```
data/
 ├── EyePACS/
 ├── Messidor2/
 └── labels.csv
```

### 4. Run Training
```bash
R1V_fundo/R1-V/src/run_grpo_vllm_qwen25vl.sh
```


---

## 🧪 Model Training Details

- **Base Framework:** [R1-V](https://github.com/StarsfieldAI/R1-V), https://github.com/EvolvingLMMs-Lab/open-r1-multimodal
- **Fine-Tuned For:** Fundus Image Reasoning and Diabetic Retinopathy Detection  
- **Training Strategy:**  
     GRPO Reinforcement Fine-Tuning  
- **Datasets:** EyePACS, Messidor-2, FundusDR, and internal curated datasets.  

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|--------------|
| **AUC (ROC)** | Overall classification discrimination |
| **QWK** | Quadratic weighted kappa for ordinal grading |
| **Sensitivity / Specificity** | Clinical performance indicators |

---

## 📜 Acknowledgements

This project builds upon and extends the open-source work of  
👉 **[StarsfieldAI/R1-V](https://github.com/StarsfieldAI/R1-V)**  
👉 **https://github.com/EvolvingLMMs-Lab/open-r1-multimodal** 

We gratefully acknowledge their contribution to the open research community in developing the R1-V framework, which inspired and enabled the multimodal reasoning architecture used in **R1V_fundo**.

---

## 📄 License

This project follows the same license terms as the original R1-V repository unless otherwise specified.  



---
