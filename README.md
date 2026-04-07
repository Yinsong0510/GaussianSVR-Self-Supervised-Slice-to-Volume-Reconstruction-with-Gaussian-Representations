# GaussianSVR

**Self-Supervised Slice-to-Volume Reconstruction with Gaussian Representations for Fetal MRI**

Yinsong Wang, Thomas Fletcher, Xinzhe Luo, Aine Travers Dineen, Rhodri Cusack, Chen Qin

Department of Electrical and Electronic Engineering and I-X, Imperial College London \
Trinity College Institute of Neuroscience, Trinity College Dublin

---

GaussianSVR is a self-supervised framework for reconstructing high-resolution 3D fetal brain MRI volumes from motion-corrupted stacks of 2D slices. It represents the target volume using 3D Gaussian primitives and jointly optimizes Gaussian parameters and slice-wise rigid transformations through a simulated forward slice acquisition model, eliminating the need for ground-truth supervision.

![Framework](docs/framework.png)

## Key Features

- **3D Gaussian Representation**: Models the 3D volume as a set of Gaussian representations with learnable position, scaling, rotation, and intensity. Enables spatially localized, fine-grained adaptation to complex fetal brain anatomy.
- **Self-Supervised Training**: Uses a differentiable forward slice acquisition model to re-simulate acquired slices from the reconstructed volume, optimizing via L1 + D-SSIM + TV loss without any ground-truth volume.
- **Multi-Resolution Strategy**: Coarse-to-fine optimization — rigid motion is first estimated at low resolution for stability, then refined at full resolution for anatomical detail.
- **Joint Optimization**: Simultaneously optimizes Gaussian parameters and per-slice rigid transforms, with adaptive densification (clone/split/prune) of Gaussians during training.
- **Pretrained SVoRT Initialization**: Uses the [NeSVoR](https://github.com/daviddmc/NeSVoR) pretrained SVoRTv2 model (auto-downloaded from Zenodo) for initial motion estimation.

## Installation

### Prerequisites

- Python >= 3.9
- CUDA >= 11.8
- PyTorch >= 2.0

### Setup

```bash
git clone https://github.com/YinsongWang/GaussianSVR.git
cd GaussianSVR

conda create -n gsvr python=3.9
conda activate gsvr

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install nibabel scipy pyyaml kornia pytorch-lightning tensorboard
```

CUDA extensions for transform conversion, slice acquisition, and Gaussian intensity computation are compiled automatically on first run via PyTorch's JIT.

## Usage

### Reconstruct from MRI Stacks

```bash
cd src
python reconstruct_nesvor.py \
    --stacks /path/to/stack1.nii.gz /path/to/stack2.nii.gz /path/to/stack3.nii.gz \
    --slice_thickness 3.0 \
    --output ../output/
```

**Arguments:**
| Argument | Description | Default |
|---|---|---|
| `--stacks` | Paths to NIfTI stack files (at least 2 recommended) | Required |
| `--slice_thickness` | Slice thickness / inter-slice gap in mm | Required |
| `--config` | Path to YAML config file | `config/default.yaml` |
| `--output` | Output directory | `output` |

**Outputs:**
- `volume_svort_init.nii.gz` — Initial volume from SVoRT
- `volume_reconstructed.nii.gz` — Final Gaussian-reconstructed volume
- `transforms.pt` — Optimized per-slice rigid transforms
- `checkpoints/` — Gaussian model snapshots

## Data Format

Input stacks should be 3D NIfTI files (`.nii` or `.nii.gz`) where:
- Each file is one stack of 2D slices
- The third dimension indexes slices
- Multiple stacks from different scan orientations (e.g., axial, coronal, sagittal) are recommended for robust reconstruction

The pipeline automatically:
1. Reads voxel spacing and affine from the NIfTI header
2. Resamples to 1.0 mm in-plane resolution
3. Crops/pads to 128x128 for SVoRT
4. Maps SVoRT outputs back to the original geometry

## Experimental Results

Evaluated on the [FeTA dataset](https://doi.org/10.7303/syn25649159) with simulated motion (3 orthogonal stacks, 15-30 slices each, 1 mm in-plane, 2.5-3.5 mm thickness):

| Method | PSNR (dB) | SSIM | NRMSE |
|--------|-----------|------|-------|
| NiftyMIC | 21.17 (1.95) | 0.7653 (0.0559) | 0.0989 (0.0234) |
| SVoRT | 23.98 (2.65) | 0.8209 (0.0618) | 0.0905 (0.1227) |
| NeSVoR | 25.58 (1.81) | 0.8940 (0.0407) | 0.0536 (0.0105) |
| **GaussianSVR** | **28.19 (3.02)** | **0.9281 (0.0552)** | **0.0468 (0.0219)** |

## Citation

```bibtex
@inproceedings{wang2025gaussiansvr,
  title={Self-Supervised Slice-to-Volume Reconstruction with Gaussian Representations for Fetal MRI},
  author={Wang, Yinsong and Fletcher, Thomas and Luo, Xinzhe and Dineen, {\'A}ine Travers and Cusack, Rhodri and Qin, Chen},
  booktitle={IEEE International Symposium on Biomedical Imaging (ISBI)},
  year={2026}
}
```

## Acknowledgements

- [NeSVoR](https://github.com/daviddmc/NeSVoR) — Pretrained SVoRTv2 model and transform utilities
- [SVoRT](https://github.com/daviddmc/SVoRT) — Iterative transformer for slice-to-volume registration
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — Original Gaussian splatting framework

## License

This project is for research purposes. Please contact the authors for licensing information.

