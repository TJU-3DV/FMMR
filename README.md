# FMMR: Few-view Multi-person Mesh Reconstruction

The official code for **FMMR** — reconstructing multiple SMPL bodies from uncalibrated multi-view RGB images.

Authors: <!-- TODO: author list -->

[\[Project\]](#) | [\[Paper\]](#)

![figure](assets/Pipeline.png)

*Overview of FMMR. (a) Human-aware unified 3D feature space initialization from sparse uncalibrated multi-view RGB. (b) Spatial contrastive learning across 3D and multi-view 2D cues. (c) 3D-aware human pose and shape reasoning with an SMPL regressor.*

## Installation

Create a conda environment and install dependencies for FMMR.

```bash
conda create -n fmmr python=3.10
conda activate fmmr

pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-demo.txt

# CUDA extensions (gsplat / torch-scatter / pytorch3d); depends on your CUDA toolchain
bash scripts/install_cuda_deps.sh

pip install -e . --no-deps
pip install -r third_party/dinov3/requirements.txt
pip install -e third_party/dinov3 --no-deps
```

Inference requires an NVIDIA GPU with CUDA and bfloat16 support. More install details are in [README.detailed.md](README.detailed.md).

## Getting Started

**Step1:**  
Download the official SMPL model from the [SMPL](https://smpl.is.tue.mpg.de/) / SMPLify website and put the files under `ckpt/smpl/models/`:

```text
ckpt/smpl/models/
  SMPL_NEUTRAL.pkl
  SMPL_MALE.pkl
  SMPL_FEMALE.pkl
```

**Step2:**  
Download trained models <!-- TODO: add download URL --> and put them in `ckpt`. You also need DINOv3 and ROMP weights. The directory should look like:

```text
ckpt/
  fmmr_pipeline.pth
  dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
  romp/ROMP.pkl
  romp/SMPL_NEUTRAL.pth
  smpl/models/ ...
```

`fmmr_pipeline.pth` merges Human Recognition, Semantic Embedding, and the multi-person SMPL Decoder (including the VGGT backbone).

**Step3:**  
Organize multi-view images of the same scene at the same time (JPG/PNG; at least 2 views, up to 6 used by default):

```text
input/0000/
  cam01.jpg
  cam02.jpg
  cam03.jpg
```

Run demo:

```bash
python test.py --inputs ./input/0000 --output-dir ./output/0000 --device cuda:0
```

Results include the background point cloud, per-person and combined SMPL meshes, SMPL parameters, predicted cameras, and mesh reprojection overlays. Keep `smpl_all_world.obj` together with `smpl_all_world.mtl` to display person colors.

```bash
python test.py --help
```

You can also load the pipeline in Python:

```python
from lib.network.fmmr import FMMRPipeline

pipeline = FMMRPipeline.from_pretrained("ckpt/fmmr_pipeline.pth", device="cuda:0")
```

## Visualization

![figure](assets/demo_scene_group02.png)

*3D scene reconstruction (Gaussian / point cloud + SMPL) from `output/test_output_old/group02`.*

<p float="left">
  <img src="assets/demo_reprojections/view_00_cam0_cam01_00381_overlay.png" width="32%" />
  <img src="assets/demo_reprojections/view_01_cam1_cam02_00381_overlay.png" width="32%" />
  <img src="assets/demo_reprojections/view_02_cam2_cam03_00381_overlay.png" width="32%" />
</p>
<p float="left">
  <img src="assets/demo_reprojections/view_03_cam3_cam05_00381_overlay.png" width="32%" />
  <img src="assets/demo_reprojections/view_04_cam4_cam06_00381_overlay.png" width="32%" />
  <img src="assets/demo_reprojections/view_05_cam5_cam08_00381_overlay.png" width="32%" />
</p>

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{FMMR,
  author = {},
  title = {FMMR: Few-view Multi-person Mesh Reconstruction},
  year = {},
  url = {}
}
```

## Acknowledgments

Some of the code are based on the following works. We gratefully appreciate the impact it has on our work.

[VGGT](https://github.com/facebookresearch/vggt)  
[DINOv3](https://github.com/facebookresearch/dinov3)  
[ROMP](https://github.com/Arthur151/ROMP)  
[SMPL](https://smpl.is.tue.mpg.de/)

## License

VGGT and DINOv3 retain their respective licenses; see [LICENSE.txt](LICENSE.txt) and [third_party/dinov3/LICENSE.md](third_party/dinov3/LICENSE.md). Model assets and datasets follow their own terms.

<!-- TODO: Specify the license for the original FMMR code. -->
