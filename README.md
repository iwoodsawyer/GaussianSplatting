# 3D Gaussian Splatting in MATLAB

[![View on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://www.mathworks.com/matlabcentral/fileexchange)

A minimal MATLAB implementation of **3D Gaussian Splatting (3DGS)** for novel-view synthesis, based on the formulation of Kerbl et al. (2023). It trains anisotropic 3D Gaussians from a COLMAP structure-from-motion reconstruction and renders images with a vectorized rasterizer. GPU acceleration is used when available, with CPU fallback.

[View the published training script and results](html/trainGaussianSplat.html).

## Highlights

- **Pure MATLAB** — no MEX files or external dependencies; COLMAP binary files (`cameras.bin`, `images.bin`, `points3D.bin`) are parsed natively.
- **Vectorized rendering** — projection and covariance computation for all Gaussians × all batch blocks in a single pass (`[N x B]` arrays), followed by depth-sorted, chunked alpha compositing with early termination.
- **Standard 3DGS covariance projection** — screen-space covariance via `Sigma_2D = J * Rcw * R S Sᵀ Rᵀ * Rcwᵀ * Jᵀ` with low-pass filtering, matching the reference implementation.
- **Block-based training** — images are streamed as blocks with an overlap halo through `blockedImageDatastore`; the block size is selected automatically to reduce padding.
- **Multi-resolution schedule** — training starts at half resolution and switches to full resolution at about one-third of the training epochs.
- **Adaptive densification** — the active Gaussian count grows from `initNumGaussians` toward `maxNumGaussians`; later events clone or split high-gradient Gaussians and prune low-contribution ones.
- **L1 + SSIM loss** with a custom training loop, Adam optimizer, and live `trainingProgressMonitor` preview.

## Requirements

- MATLAB R2023b or newer (recommended)
- [Deep Learning Toolbox](https://www.mathworks.com/products/deep-learning.html) (`dlarray`, `minibatchqueue`, `trainingProgressMonitor`)
- [Image Processing Toolbox](https://www.mathworks.com/products/image-processing.html) (`blockedImage`, `blockedImageDatastore`)
- [Parallel Computing Toolbox](https://www.mathworks.com/products/parallel-computing.html) for GPU acceleration (`gpuArray`); without a usable GPU, the implementation falls back to CPU
- A CUDA-capable NVIDIA GPU is recommended for practical training times; memory requirements depend on block size, batch size, and Gaussian count

## Getting Started

### 1. Get a dataset

Download the Tanks & Temples / Deep Blending example datasets from the original 3DGS project:

<https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip>

Extract it, e.g. to `C:\Source\tandt_db`. Each scene folder (such as `tandt\train`) contains the input images plus the COLMAP sparse reconstruction (`sparse/0` with `cameras.bin`, `images.bin`, `points3D.bin`). Any dataset in this COLMAP layout works.

### 2. Configure and train

Open `trainGaussianSplat.m` and set the dataset path. The dataset folder must contain an `images` directory and a COLMAP reconstruction at `sparse/0` (`cameras.bin`, `images.bin`, and `points3D.bin`). The current example settings are:

```matlab
datasetPath      = 'C:\Source\tandt_db\tandt\train'; % Update this path
initNumGaussians = 2000;   % Active Gaussians at the start of training
maxNumGaussians  = 8000;   % Upper bound reached via densification
numImages        = 301;    % Number of images in the dataset
blockSize        = [];     % Automatically selected when empty
overlapSize      = [5, 5]; % Context halo around each block
```

Then run the script:

```matlab
trainGaussianSplat
```

A live training monitor shows the loss curve and a rendered preview block. After training, the script stitches blocks into full-image prediction/ground-truth comparisons. More Gaussians produce sharper renders at the cost of memory and training time.

A pretrained parameter set is included in `gaussians.mat`. Set `doTraining = false` to skip optimization and use these parameters for the preview. The dataset path is still required because the preview renders and compares against images from that scene.

### 3. Load COLMAP data standalone

The COLMAP reader can also be used on its own:

```matlab
[cameras, images, points] = ColmapLoader.load('path/to/sparse/0');
plot3([points.x], [points.y], [points.z], '.');  % view the sparse point cloud
```

### 4. View trained Gaussians as a point cloud

Run `create3Dpoints` to inspect the parameters in `gaussians.mat`. By default,
`cloudMode = "centres"` shows one point per retained Gaussian without
downsampling, and `colorMode = "dc"` uses view-independent spherical-harmonic
DC colors. The initial view focuses on the 5th--95th position percentiles
with Y pointing down (COLMAP convention). No points are removed by focusing;
set `focusPercentiles = []` to display the full extent.

Set `cloudMode = "sampled"` for the original batched densification. Adjust
`pointsPerSplat` (sample density per volume times opacity), `batchSize`
(Gaussians per batch), and `gridStep` (downsampled resolution). Large background
Gaussians can dominate volume-weighted sampling and obscure the subject; coarse
downsampling can also erase small-scale detail. Peak memory depends on sampled
point count, not just Gaussian count. Counts are accumulated in double precision
to keep sample rows and Gaussian indices aligned above 2^24 points.

`colorMode = "sh"` evaluates view-dependent colors at `cameraPosition`
(world coordinates, default `[0 0 0]`); colors do not update when rotating
the viewer. A point cloud does **not** reproduce a Gaussian render's
transparency, depth compositing, or view-dependent appearance. Use the training
script's render preview for that comparison.

To use the camera pose of a COLMAP image, set `datasetPath` to the dataset root
and `imageId` to that image's COLMAP image ID in `create3Dpoints`. The script
prints the camera centre in world coordinates and uses the image's position,
forward direction, and image-up direction for the viewer and SH colors. This
is the ID stored in `images.bin`, not the image's position in a sorted list or
the number in its filename. Leave `imageId = []` to use the configured
`cameraPosition` for SH colors and MATLAB's default viewer camera.

The viewer uses Computer Vision Toolbox (`pointCloud`, `pcshow`); sampled mode
also uses `pcdownsample` and `pccat`. Statistics and Machine Learning Toolbox
is needed for percentile focusing (`prctile`) and sampling (`lhsdesign`,
`norminv`).

## Repository Structure

| File | Description |
| --- | --- |
| `trainGaussianSplat.m` | Main training script: options, custom training loop, densification schedule, live preview |
| `GaussianSplatter.m` | Core class: learnable Gaussian parameters, GPU-vectorized projection/culling, tile-based rasterizer, L1+SSIM loss, densification |
| `ColmapData.m` | Dataset wrapper: builds the blocked image / per-block camera datastores at two resolution levels, initializes Gaussians from SfM points |
| `ColmapLoader.m` | Reader for COLMAP binary exports (`cameras.bin`, `images.bin`, `points3D.bin`) |
| `create3Dpoints.m` | Focused Gaussian-centre inspection or batched sampling into a colored point cloud |
| `gaussians.mat` | Pretrained Gaussians for the example dataset |
| [`html/trainGaussianSplat.html`](html/trainGaussianSplat.html) | Published training script and results |

`GaussianSplatter01.m`, `GaussianSplatter02.m`, the `test*.m` scripts, and the `.asv` files are development experiments or backups; the supported training entry point is `trainGaussianSplat.m` with `GaussianSplatter.m`.

## How It Works

1. **Initialization** — Gaussians are seeded from the COLMAP sparse point cloud: positions from the 3D points, colors converted to spherical-harmonic DC terms, scales from nearest-neighbor distances, identity rotations, and initial opacities.
2. **Projection** — each training step transforms Gaussians into camera space, computes the 2D screen-space covariance via the affine Jacobian, and culls Gaussians outside each image block. Camera intrinsics are adjusted to the block origin.
3. **Rasterization** — visible Gaussians are depth-sorted and alpha-composited front-to-back in chunks with a saturation-based early exit.
4. **Optimization** — each rendered block is compared with its ground truth using L1 + SSIM loss; `dlgradient` and Adam update positions, scales, rotations, opacities, and SH colors.
5. **Densification** — the model grows the active Gaussian pool from the COLMAP point cloud, then clones/splits high-gradient Gaussians and prunes low-contribution ones.

## Reference

> Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis.
> *3D Gaussian Splatting for Real-Time Radiance Field Rendering.*
> ACM Transactions on Graphics (SIGGRAPH 2023).
> <https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/>

