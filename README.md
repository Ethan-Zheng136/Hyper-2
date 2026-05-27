# Hyper² : Unleashing Hyperbolic Geometry's Full Potential via Dual-Space Consistency

Official PyTorch implementation of **Hyper²**, a point-cloud-completion
framework that places the same `arcosh(1 + α·d²)` non-linearity at
**both** ends of the network — as a positional bias inside the
refinement attention **and** as the Chamfer training loss — under a
single shared curvature `α`. Closing this *cross-geometry mismatch*
delivers large gains over hyperbolic-loss-only methods at ~1.6 % extra
FLOPs.

<p align="center">
  <img src="teaser.png" width="90%" alt="Hyper² teaser"/>
</p>

## Framework

<p align="center">
  <img src="assets/overview.png" width="92%" alt="Hyper² framework overview"/>
</p>

Hyper² introduces two surgical changes over a Euclidean completion
backbone: (i) a **hyperbolic distance encoding** that injects
`arcosh(1+α·d²)` as a positional bias on the refinement attention, and
(ii) a **hyperbolic Chamfer loss** under the same `α`. Both operators
are scalar non-linearities on Euclidean distances over a standard
`O(N log N)` KNN graph — no manifold projection, no Poincaré-ball
machinery.

## Main Results

ShapeNet-55 (ℓ² CD × 10³ ↓, F1@1 % ↑):

| Method | CD-S | CD-M | CD-H | **CD-Avg ↓** | F1 ↑ |
| :-- | :--: | :--: | :--: | :--: | :--: |
| PoinTr | 0.58 | 0.88 | 1.79 | 1.09 | 0.464 |
| SeedFormer | 0.50 | 0.77 | 1.49 | 0.92 | 0.472 |
| SVDFormer | 0.48 | 0.70 | 1.30 | 0.83 | 0.451 |
| **Hyper² (Ours)** | **0.37** | **0.55** | **1.01** | **0.64** | **0.523** |

Summary across benchmarks:

| Benchmark | Metric | Improvement vs. SVDFormer |
| :-- | :-- | :-- |
| ShapeNet-55 | ℓ² CD-Avg | **−22.9 %** |
| ShapeNet-34 (21 unseen) | ℓ² CD-Avg | **−37.5 %** |
| PCN | ℓ¹ CD-Avg | **−2.8 %** |
| KITTI | Fidelity / MMD | **≈2× / −25 %** |
| Compute | FLOPs | **+1.6 %** |

## Getting Started

### Environment

```bash
git clone https://github.com/Ethan-Zheng136/Hyper-2.git
cd Hyper-2

conda create -n hyper2 python=3.8 -y
conda activate hyper2

pip install torch torchvision           # CUDA build matching your driver
pip install numpy easydict tqdm matplotlib opencv-python h5py munch open3d transforms3d
```

Build the CUDA extensions:

```bash
cd metrics/CD/chamfer3D && python setup.py install --user && cd ../../../
cd metrics/EMD            && python setup.py install --user && cd ../../
cd pointnet2_ops_lib      && python setup.py install --user && cd ../
```

### Datasets

Edit `*_POINTS_PATH` and `CONST.WEIGHTS` in `config_55.py` /
`config_pcn.py` to point at your local copy of **ShapeNet-55 / 34**
and **PCN**. Split files live under `datasets/`.

### Train

```bash
bash train_pcn.sh        # PCN
bash train_55.sh         # ShapeNet-55 / 34
```

### Test

Point `cfg.CONST.WEIGHTS` at your checkpoint, then:

```bash
bash test_pcn.sh
bash test_55.sh
```

The shared curvature `α = 0.5` is set inside the hyperbolic ops:
`models/SVDFormer.py` (`hyper_cd_distance`, `arcosh`) and
`utils/loss_utils.py` (`calc_cd_like_hyperV2`, `arcosh`). Change both
sites together if you sweep it.

## Acknowledgements

This codebase builds on **SVDFormer** (ICCV 2023) and the hyperbolic
Chamfer formulation of **HyperbolicCD** (ICCV 2023). We thank the
authors for releasing their code.

## Citation

```bibtex
@inproceedings{hyper2_bmvc,
  title     = {Hyper$^2$: Unleashing Hyperbolic Geometry's Full Potential via Dual-Space Consistency},
  author    = {Zheng, Guantian and others},
  booktitle = {Proceedings of the British Machine Vision Conference (BMVC)},
  year      = {2026}
}
```

## License

Released under the [MIT License](LICENSE).
