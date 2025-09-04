# EfficientNetV2-S (PyTorch) – Tiny CIFAR-10 Experiment

## 📖 Overview
This project trains an **EfficientNetV2-S** classifier on a **tiny balanced subset** of CIFAR-10 (≤100 total images across selected classes).  
It saves the best model checkpoint and generates training/validation plots.

**Outputs:**
- `plots/loss.png` → Loss per epoch  
- `plots/accuracy.png` → Accuracy per epoch  
- `checkpoints/best_efficientnet_v2_s.pt` → Best model weights  

---

## ⚙️ Setup (Windows / Python 3.12)

```powershell
# install dependencies
py -3.12 -m pip install -r requirements.txt
🚀 How to Run:

powershell:
cd experiments/efficientnetv2_pytorch_mini
▶️ Example: Train on 5 CIFAR-10 classes with 100 total images
powershell
py -3.12 -u train.py --classes automobile truck ship cat dog --max_total 100 --epochs 12
⚡ Optional: Freeze EfficientNet backbone (faster training on tiny datasets)
py -3.12 -u train.py --freeze_backbone

📂 Project Structure
model.py – EfficientNetV2-S head + options (freeze, param count)

train.py – data loading (tiny CIFAR-10), train/val/test loops, plots

tests/ – minimal unit tests for model & data (optional but recommended)

checkpoints/ (ignored) – best model and metadata

plots/ (ignored) – loss/accuracy images

📝 Notes
Code is modular and easy to extend (e.g., ImageFolder loader for large datasets).

Supports optional backbone freezing for faster training on tiny datasets.

Outputs include model checkpoints and training plots for loss/accuracy.