<p align="center">
  <h1 align="center">GaussianSVR</h1>
  <h2 align="center">Self-Supervised Slice-to-Volume Reconstruction<br>with Gaussian Representations for Fetal MRI (ISBI 2026 Oral)</h2>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2601.22990"><img src="https://img.shields.io/badge/arXiv-2601.22990-b31b1b.svg" alt="arXiv"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-%3E%3D3.9-3776AB.svg?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%3E%3D2.0-EE4C2C.svg?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-%3E%3D11.8-76B900.svg?logo=nvidia&logoColor=white" alt="CUDA"></a>
  <a href="https://github.com/YinsongWang/GaussianSVR/stargazers"><img src="https://img.shields.io/github/stars/YinsongWang/GaussianSVR?style=social" alt="Stars"></a>
</p>

<p align="center">
  <b>Yinsong Wang<sup>1</sup> &middot; Thomas Fletcher<sup>1</sup> &middot; Xinzhe Luo<sup>1</sup> &middot; Aine Travers Dineen<sup>2</sup> &middot; Rhodri Cusack<sup>2</sup> &middot; Chen Qin<sup>1</sup></b><br>
  <sup>1</sup>Department of Electrical and Electronic Engineering and I-X, Imperial College London<br>
  <sup>2</sup>Trinity College Institute of Neuroscience, Trinity College Dublin
</p>

<p align="center">
  <a href="#-highlights">Highlights</a> •
  <a href="#%EF%B8%8F-framework">Framework</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-data-format">Data</a> •
  <a href="#-results">Results</a> •
  <a href="#-citation">Citation</a>
</p>

---

## ✨ Highlights

- 🎯 **3D Gaussian Representation** — Models the volume as a set of Gaussian primitives with learnable position, scaling, rotation, and intensity for spatially localized, fine-grained adaptation to complex fetal brain anatomy
- 🔁 **Self-Supervised Training** — A differentiable forward slice acquisition model re-simulates acquired slices from the reconstructed volume; optimized with L1 + D-SSIM + TV loss, **no ground-truth volume required**
- 📶 **Multi-Resolution Strategy** — Coarse-to-fine optimization: rigid motion is first estimated at low resolution for stability, then refined at full resolution for anatomical detail
- ⚡ **Joint Optimization** — Simultaneously optimizes Gaussian parameters and per-slice rigid transforms, with adaptive densification (clone/split/prune) of Gaussians during training
---

## 🏗️ Framework

<p align="center">
  <img src="docs/framework.png" width="92%"/>
</p>

---

## 📦 Installation

### Prerequisites

| Requirement | Version |
|:------------|:--------|
| Python | ≥ 3.9 |
| CUDA | ≥ 11.8 |
| PyTorch | ≥ 2.0 |

> [!NOTE]
> CUDA extensions for transform conversion, slice acquisition, and Gaussian intensity computation are compiled automatically on first run via PyTorch's JIT.

---

## 🚀 Usage

### Reconstruct from MRI Stacks

```bash
cd src
python reconstruct_nesvor.py \
    --stacks /path/to/stack1.nii.gz /path/to/stack2.nii.gz /path/to/stack3.nii.gz \
    --slice_thickness 3.0 \
    --output ../output/
```

### Arguments

| Argument | Description | Default |
|:---------|:------------|:--------|
| `--stacks` | Paths to NIfTI stack files (≥ 2 recommended) | **Required** |
| `--slice_thickness` | Slice thickness / inter-slice gap in mm | **Required** |
| `--config` | Path to YAML config file | `config/default.yaml` |
| `--output` | Output directory | `output` |

### Outputs

| File | Description |
|:-----|:------------|
| 📄 `volume_svort_init.nii.gz` | Initial volume from SVoRT |
| 📄 `volume_reconstructed.nii.gz` | Final Gaussian-reconstructed volume |
| 💾 `transforms.pt` | Optimized per-slice rigid transforms |
| 📁 `checkpoints/` | Gaussian model snapshots |

---

## 📂 Data Format

Input stacks should be 3D NIfTI files (`.nii` or `.nii.gz`) where:

- 🧩 Each file is one stack of 2D slices
- 📐 The third dimension indexes slices
- 🔄 Multiple stacks from different scan orientations (e.g., axial, coronal, sagittal) are recommended for robust reconstruction

<details>
<summary><b>📋 Automatic Preprocessing Pipeline</b></summary>
<br>

The pipeline automatically:

1. Reads voxel spacing and affine from the NIfTI header
2. Resamples to 1.0 mm in-plane resolution
3. Crops / pads to 128 × 128 for SVoRT
4. Maps SVoRT outputs back to the original geometry

</details>

> [!TIP]
> For best results, provide at least 3 stacks acquired in orthogonal orientations.

---

## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{wang2026self,
  title={Self-Supervised Slice-to-Volume Reconstruction with Gaussian Representations for Fetal MRI},
  author={Wang, Yinsong and Fletcher, Thomas and Luo, Xinzhe and Dineen, Aine Travers and Cusack, Rhodri and Qin, Chen},
  journal={arXiv preprint arXiv:2601.22990},
  year={2026}
}
```

---

## 🙏 Acknowledgements

- 🔧 [**NeSVoR**](https://github.com/daviddmc/NeSVoR) — Pretrained SVoRTv2 model and transform utilities
- 🧠 [**SVoRT**](https://github.com/daviddmc/SVoRT) — Iterative transformer for slice-to-volume registration
- 💎 [**3D Gaussian Splatting**](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — Original Gaussian splatting framework

---

## 📄 License

This project is for **research purposes**. Please contact the authors for licensing information.

---

<p align="center">
  📧 <b>Contact:</b> <a href="mailto:y.wang23@imperial.ac.uk">y.wang23@imperial.ac.uk</a>
</p>
