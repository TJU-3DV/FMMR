# FMMR: Feed-Forward Multi-view Multi-person Reconstruction with Contrastive Human-Aware 3D Representation

**News**

<!-- TODO: Add release dates and announcements. -->

### Project Page: <!-- TODO: URL --> | [Paper](https://doi.org/10.1007/s11263-026-03000-0)

Official repository for **Feed-Forward Multi-view Multi-person Reconstruction with Contrastive Human-Aware 3D Representation** (IJCV 2026). This repository provides inference code for reconstructing multiple SMPL bodies from uncalibrated multi-view RGB images. Any discussions or questions are welcome!

## Dependencies

Our inference environment uses Python 3.10, PyTorch 2.1.0 with CUDA 12.1, and PyTorch3D. Run the following commands from the project root:

```bash
conda create -n fmmr python=3.10
conda activate fmmr

pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-demo.txt

# Install CUDA extensions compatible with your PyTorch and CUDA versions.
bash scripts/install_cuda_deps.sh

pip install -e . --no-deps
pip install -r third_party/dinov3/requirements.txt
pip install -e third_party/dinov3 --no-deps
```

The inference pipeline requires an NVIDIA GPU with CUDA and bfloat16 support. The CUDA extension installation procedure depends on your environment.

## Pretrained model

Download link: <!-- TODO: add the FMMR checkpoint download URL. -->

Put the FMMR checkpoint and the required external models in `./ckpt`. They should be organized as shown below:

```text
FMMR
├─lib
└─ckpt
    ├─fmmr_pipeline.pth
    ├─dinov3
    │   └─dinov3_vits16_pretrain_lvd1689m-08c60483.pth
    ├─romp
    │   ├─ROMP.pkl
    │   └─SMPL_NEUTRAL.pth
    └─smpl
        └─models
            ├─SMPL_NEUTRAL.pkl
            ├─SMPL_MALE.pkl
            └─SMPL_FEMALE.pkl
```

`fmmr_pipeline.pth` combines Human Recognition, Semantic Embedding, and the multi-person SMPL Decoder, including the VGGT backbone used for inference. ROMP, DINOv3, and the SMPL body models are loaded separately from the directories above.

<!-- TODO: Add download links and preparation instructions for external model assets. -->

## Testing

The input images should capture the same scene at the same instant from different viewpoints. JPG and PNG images are supported, and resizing is handled automatically. By default, inference requires at least two views and selects up to six views. Organize the images as shown below:

```text
FMMR
├─lib
└─input
    └─0000
        ├─cam01.jpg
        ├─cam02.jpg
        └─cam03.jpg
```

You can test it by running the following command from the project root. The results are saved in `./output/0000/`:

```bash
python test.py --inputs ./input/0000 --output-dir ./output/0000 --device cuda:0
```

The results include the background point cloud, individual and combined SMPL meshes, SMPL parameters, predicted cameras, and mesh reprojection overlays. Keep `smpl_all_world.obj` and `smpl_all_world.mtl` together to display the person colors. Reprojection overlays are rendered with PyTorch3D lighting and smooth normals.

The combined model can also be loaded in Python:

```python
from lib.network.fmmr import FMMRPipeline

pipeline = FMMRPipeline.from_pretrained("ckpt/fmmr_pipeline.pth", device="cuda:0")
```

Run `python test.py --help` for additional options.

## Citation

If you find this work useful for your research, please use the following BibTeX entry.

<!-- TODO: Complete the publication type, authors, title, venue, year, and URL before using this citation. -->

```bibtex
@misc{FMMR,
  author = {},
  title = {FMMR: Few-view Multi-person Mesh Reconstruction},
  year = {},
  url = {}
}
```

## License

VGGT and DINOv3 retain their respective licenses; see [LICENSE.txt](LICENSE.txt) and [third_party/dinov3/LICENSE.md](third_party/dinov3/LICENSE.md). Model assets and datasets are subject to their respective terms.

<!-- TODO: Specify the license for the original FMMR code. -->
