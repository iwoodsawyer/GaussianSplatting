classdef GaussianSplatter < handle
    % GaussianSplatter - GPU-Optimized 3D Gaussian Splatting
    % MATLAB implementation of 3D Gaussian Splatting training loop with
    % high-performance GPU acceleration and minimal CPU/GPU synchronization.
    %
    % Covariance Projection (Kerbl et al. 2023):
    %   The 2D screen-space covariance is computed via the standard 3DGS
    %   formulation rather than an ad-hoc 3D SVD:
    %
    %     Sigma_world = R * S * S^T * R^T          (3x3, world space)
    %     Sigma_cam   = Rcw * Sigma_world * Rcw^T  (3x3, camera space)
    %     J = [fx/z, 0,    -fx*x/z^2]              (2x3 affine Jacobian)
    %         [0,    fy/z, -fy*y/z^2]
    %     Sigma_2D = J * Sigma_cam * J^T           (2x2, screen space)
    %     Sigma_2D += 0.3 * I_2                    (low-pass filter)
    %     Sigma_2D_inv = inv(Sigma_2D)             (2x2, for falloff)
    %
    %   The Gaussian falloff in createImage becomes:
    %     d^T * Sigma_2D_inv * d   (standard Mahalanobis distance, 2D)
    %
    %   This replaces the previous approach of rotating the 3D ellipsoid,
    %   scaling all axes by fx only (ignoring fy), running a custom 3D
    %   Jacobi SVD, and using U(1:2,:,k)/S(:,:,k) for the falloff.
    %   The new formulation is cheaper (2x2 inversion vs 3D SVD), correct
    %   (uses both fx and fy), and matches the reference implementation.
    %
    % Tile-Aware Rasterization:
    %   this.data.images (ColmapData) now yields one blockedImageDatastore
    %   block per read(), sized blockSize + 2*overlap, with camera cx/cy
    %   already shifted to that block's local origin. blockSize therefore
    %   drives BOTH the dataset's block grid and createImage's internal
    %   tile-culling loop below — each Gaussian is culled to only the tiles
    %   its bounding box overlaps within the current block canvas.
    %
    %   Tile culling is performed ONCE on CPU per batch (via
    %   buildTileList), then rasterizeToGPU processes pre-built tile lists
    %   without any extractdata() calls in the inner loop.
    %
    %   Because the loss (L1 + SSIM) is computed on this.image/this.image_gt
    %   directly, and those are now block-sized, the loss is automatically
    %   scoped to the block instead of the full image — no separate code
    %   path is needed for that.
    %
    % Resolution Schedule:
    %   Training starts at half resolution (fast early convergence).
    %   At levelSwitchEpoch, updateResolution() reallocates GPU buffers
    %   (image, X_vec, Y_vec, T) to the full-resolution canvas dimensions.
    %   Without this reallocation the GPU buffers would remain sized to
    %   the half-resolution canvas, clipping all Gaussian projections.
    %
    % GPU Memory Strategy:
    %   initStorage() unconditionally allocates all working arrays on GPU.
    %   Projection buffers are allocated ONCE and reused across all iterations,
    %   eliminating fragmentation and per-batch sync/allocation overhead.
    %   minibatchqueue delivers data with OutputEnvironment='gpu', so GPU
    %   residency is guaranteed from the first iteration onwards.
    %
    % GPU Optimization Strategy:
    %   - Minimal extractdata() calls: only one per-batch call for CPU culling
    %   - Fused GPU operations: reduce intermediate allocations
    %   - Persistent buffers: reuse arrays across iterations (no per-batch malloc)
    %   - Separable operations: 1-D coordinates and SSIM convolutions
    %   - Numerically stable: clamped alpha compositing
    %   - Clean code: explicit meshgrid for 2-D expansion (no confusing reshapes)

    properties
        % --- Sizes ---
        numImages
        numGaussians
        miniBatchSize
        imageWidth
        imageHeight
        imageChannel

        % --- Tile Settings ---
        % Tile dimensions for the rasterizer canvas partitioning.
        % Independent of the image datastore — full images are loaded per
        % iteration and partitioned into tiles inside createImage.
        % 
        % TUNING GUIDE (hardware and dataset dependent):
        %   64×64:   RTX 4050/4060 (6GB), small scenes (<5k Gaussians)
        %            Better work distribution, higher CPU loop overhead
        %   128×128: RTX 3070+ (8-10GB), medium scenes
        %   256×256: RTX 3080+ (10GB+), large scenes (>50k Gaussians)
        %            Fewer tiles, less CPU overhead, early termination less effective
        %
        % Default: auto-selected based on GPU compute capability.
        blockSize = [];  % Will be auto-selected in constructor if not provided

        % Gaussians composited per vectorized chunk in rasterizeToGPU.
        % Larger = fewer kernel launches but more [H'×W'×K] temp memory.
        chunkSize = 64;

        % --- Data ---
        datasetPath
        data

        % --- Learnable Parameters ---
        params

        % --- Loss Weights ---
        lambda = single(0.2);

        % --- SSIM Constants ---
        windowSize = single(11);
        C1 = single(0.01^2);
        C2 = single(0.03^2);

        % --- Zeroth-order and second-order spherical harmonic coefficients ---
        shToColor = single([0.28209479177387814; ...
                            0.4886025119029199;  ...
                            0.4886025119029199;  ...
                            0.4886025119029199;  ...
                            1.0925484305920792;  ...
                            1.0925484305920792;  ...
                            1.0925484305920792;  ...
                            0.31539156525252005; ...
                            0.5462742152960396]);

        % --- Densification Thresholds ---
        sceneScale
        scaleThresh         = single(0.1);   % Relative to scene size
        forcePruneThreshold = single(-4.0);  % sigmoid(-4) ≈ 0.018 — prune near-invisible
        forceKeepThreshold  = single(2.0);   % sigmoid(2)  ≈ 0.88  — always keep opaque
        splitScaleFactor    = single(0.2);   % Scale reduction on split

        % --- Working Arrays (allocated in initStorage, resident on GPU) ---
        image     % Rendered output:  [H x W x 3 x B] dlarray 'SSCB'
        image_gt  % Ground truth:     [H x W x 3 x B] dlarray 'SSCB'
        camera    % Struct of current mini-batch camera parameters
        window_h  % Horizontal SSIM kernel: [1 x wS x C x C] plain gpuArray
        window_v  % Vertical SSIM kernel:   [wS x 1 x C x C] plain gpuArray
        X_vec     % Pixel column coordinates [1 x W] plain gpuArray (separable)
        Y_vec     % Pixel row coordinates    [H x 1] plain gpuArray (separable)
        T         % Per-pixel transmittance  [H x W x 1] plain gpuArray

        % --- Persistent Projection Buffers (reused every batch, allocated once) ---
        % These are allocated in initStorage() and never deallocated.
        % On every call to projectGaussiansWithCulling(), buffers are zeroed
        % and filled, then passed to createImage(). This eliminates per-batch
        % GPU memory allocation and fragmentation.
        u_buffer           % [numGaussians x 1 x 1 x B] dlarray 'SSCB'
        v_buffer           % [numGaussians x 1 x 1 x B] dlarray 'SSCB'
        alphas_buffer      % [numGaussians x 1 x 1 x B] dlarray 'SSCB'
        Sigma2D_inv_buffer % [numGaussians x 2 x 2 x B] dlarray 'SSSB'
        radii_u_buffer     % [numGaussians x 1 x B] plain gpuArray
        radii_v_buffer     % [numGaussians x 1 x B] plain gpuArray
        colors_buffer      % [numGaussians x 1 x 3 x B] dlarray 'SSCB'

        % --- CPU-based Tile Lists (cached per batch) ---
        % Pre-computed tile membership for each Gaussian, built once per batch
        % on CPU. Per batch element: tileList{b}{th, tw} = [Gaussian indices],
        % since each batch element is a different block/camera view.
        tileList           % cell array {1 x miniBatchSize} of {numTilesH x numTilesW}
        lastNumTilesH = -1 % Cache dimensions to detect resolution changes
        lastNumTilesW = -1

        % --- CPU Bounding-Box Caches (filled by buildTileList) ---
        % [numValid x B] CPU singles so rasterizeToGPU computes pixel bounds
        % without any per-Gaussian GPU→CPU sync.
        uminCPU
        umaxCPU
        vminCPU
        vmaxCPU
    end

    methods
        function this = GaussianSplatter(datasetPath, numGaussians, numImages, blockSize, overlap)
            % Constructor: load COLMAP data, initialise learnable parameters,
            % and create the separable SSIM Gaussian kernels.
            %
            % Args:
            %   datasetPath  (string) : Path to the COLMAP dataset root.
            %   numGaussians (int)    : Number of Gaussians to initialise.
            %   numImages    (int)    : Number of training images to load.
            %   blockSize    (1x2 int): Block/tile [H W]. Defaults to auto-select.
            %                          Drives both the dataset's blockedImage
            %                          grid (ColmapData) and the internal
            %                          tile-culling loop in createImage.
            %   overlap      (1x2 int): Border [H W] added around each block
            %                          (blockedImageDatastore BorderSize).
            %                          Defaults to [0 0].
            %
            % BLOCKSIZE AUTO-SELECTION:
            %   If blockSize is empty or not provided, tile size is automatically
            %   selected based on GPU compute capability:
            %     CC >= 8.0 (RTX 30+): 128×128
            %     CC >= 7.0 (RTX 20+):  96×96
            %     CC < 7.0  (Older):   256×256

            if nargin < 4 || isempty(blockSize)
                % Auto-select tile size based on GPU capability
                g = gpuDevice;
                if g.ComputeCapability >= 8.0  % RTX 30 series and newer
                    blockSize = [128, 128];
                elseif g.ComputeCapability >= 7.0  % RTX 20 series
                    blockSize = [96, 96];
                else  % Older GPUs
                    blockSize = [256, 256];
                end
                fprintf('Auto-selected tile size: %d×%d (GPU CC %.1f)\n', ...
                    blockSize(1), blockSize(2), g.ComputeCapability);
            end
            if nargin < 5 || isempty(overlap)
                overlap = [0, 0];
            end

            this.camera    = struct;
            this.blockSize = blockSize;

            % Store sizes
            this.datasetPath  = datasetPath;
            this.numGaussians = numGaussians;
            this.numImages    = numImages;

            % Load data as a blockedImageDatastore; each read() returns one
            % block matching one (offset-adjusted) repeated camera entry.
            this.data = ColmapData(datasetPath, numGaussians, numImages, blockSize, overlap);

            % Determine block canvas dimensions from the datastore preview.
            previewImg = preview(this.data.images);
            if iscell(previewImg)
                previewImg = previewImg{1};
            end
            [this.imageHeight, this.imageWidth, this.imageChannel] = size(previewImg);

            % Create learnable parameter struct (dlarray, CPU at this point)
            this.params = this.createLearnableParams(this.data.gaussians);

            % Create the separable Gaussian kernels for differentiable SSIM loss
             [this.window_h, this.window_v] = this.createSeparableWindow(...
                this.windowSize, this.imageChannel);

            % Scene scale drives the densification scale threshold
            temp            = preview(this.data.scene_scale);
            this.sceneScale = temp{1};
        end

        function updateResolution(this)
            % Reallocate GPU canvas buffers after a resolution switch.
            %
            % When setLevel(1) swaps the datastore from half-res to full-res,
            % imageHeight and imageWidth must be updated to the new canvas size
            % and all GPU buffers (image, X_vec, Y_vec, T) must be reallocated.
            % Without this call, buffers remain at half-resolution dimensions,
            % clipping all Gaussian projections on the larger canvas.
            %
            % Called in trainGaussianSplat.m immediately after setLevel():
            %   obj.data.setLevel(1);
            %   obj.updateResolution();

            % Re-read canvas dimensions from the now-active datastore
            previewImg = preview(this.data.images);
            if iscell(previewImg)
                previewImg = previewImg{1};
            end
            [this.imageHeight, this.imageWidth, this.imageChannel] = size(previewImg);

            % Reallocate all GPU working arrays to the new canvas dimensions.
            % initStorage uses this.imageHeight / this.imageWidth internally.
            this.initStorage(this.miniBatchSize);

            fprintf('GPU buffers reallocated to %d x %d canvas (separable coords).\n', ...
                this.imageHeight, this.imageWidth);
        end

        function initStorage(this, miniBatchSize)
            % Allocate all working GPU arrays.
            %
            % Called once at the first training iteration and again by
            % updateResolution() after a resolution switch. All arrays are
            % created directly on the GPU so there is no deferred
            % CPU-to-GPU copy.
            %
            % Projection buffers are now allocated ONCE and reused across
            % all iterations. They are zeroed at the start of each batch
            % call, then filled with new data. This eliminates per-batch
            % GPU memory allocation and fragmentation.
            %
            % Canvas dimensions (imageHeight × imageWidth) must reflect the
            % currently active datastore resolution before this is called.
            this.miniBatchSize = miniBatchSize;

            % Rendered image buffer — lives on GPU for the full training run.
            % Sized to the current block canvas (blockSize + 2*overlap).
            this.image = dlarray( ...
                gpuArray(zeros(this.imageHeight, this.imageWidth, 3, ...
                               this.miniBatchSize, 'single')), 'SSCB');

            % Pixel coordinate vectors (separable) — saves ~8 MB vs. dense
            % meshgrid. Span the current block canvas so
            % projectGaussiansWithCulling maps correctly with the per-block
            % intrinsics (cx, cy already shifted to the block's local origin).
            % 
            % Instead of [H×W×1] grids, use [1×W] and [H×1] vectors with
            % implicit broadcasting during Mahalanobis computation.
            this.X_vec = gpuArray(single(1:this.imageWidth));     % [1 x W]
            this.Y_vec = gpuArray(single((1:this.imageHeight)')); % [H x 1]

            % Per-pixel transmittance accumulator — reset to 1 per batch
            % Use plain gpuArray (not dlarray) since T is not differentiated
            this.T = gpuArray(ones(this.imageHeight, this.imageWidth, 1, 'single'));

            % SSIM loss constants on GPU (plain gpuArray, not dlarray)
            this.C1 = gpuArray(this.C1);
            this.C2 = gpuArray(this.C2);

            % Combined loss weight on GPU
            this.lambda = gpuArray(this.lambda);

            % Separable SSIM Gaussian kernels on GPU (plain gpuArray)
            this.window_h = gpuArray(this.window_h);
            this.window_v = gpuArray(this.window_v);

            % --- Allocate Persistent Projection Buffers ---
            % These are allocated once and reused across all training iterations.
            % On each call to projectGaussiansWithCulling(), buffers are zeroed
            % and refilled. This avoids per-batch GPU allocation, which can cause
            % fragmentation and GPU→CPU sync overhead.
            fprintf('Allocating persistent projection buffers (%d Gaussians)...\n', ...
                this.numGaussians);

            this.u_buffer = dlarray(gpuArray(zeros(this.numGaussians, 1, 1, ...
                miniBatchSize, 'single')), 'SSCB');
            this.v_buffer = dlarray(gpuArray(zeros(this.numGaussians, 1, 1, ...
                miniBatchSize, 'single')), 'SSCB');
            this.alphas_buffer = dlarray(gpuArray(zeros(this.numGaussians, 1, 1, ...
                miniBatchSize, 'single')), 'SSCB');
            this.Sigma2D_inv_buffer = dlarray(gpuArray(zeros(this.numGaussians, 2, 2, ...
                miniBatchSize, 'single')), 'SSSB');
            this.radii_u_buffer = gpuArray(zeros(this.numGaussians, 1, ...
                miniBatchSize, 'single'));
            this.radii_v_buffer = gpuArray(zeros(this.numGaussians, 1, ...
                miniBatchSize, 'single'));
            this.colors_buffer = dlarray(gpuArray(zeros(this.numGaussians, 1, 3, ...
                miniBatchSize, 'single')), 'SSCB');

            % --- Move all learnable parameters to GPU ---
            % Unconditional: minibatchqueue always delivers with
            % OutputEnvironment='gpu' so GPU residency is guaranteed.
            fprintf('Moving learnable parameters to GPU...\n');
            this.params.pws        = gpuArray(this.params.pws);
            this.params.shs        = gpuArray(this.params.shs);
            this.params.scales_raw = gpuArray(this.params.scales_raw);
            this.params.alphas_raw = gpuArray(this.params.alphas_raw);
            this.params.rots_raw   = gpuArray(this.params.rots_raw);
        end

        function [loss, grads] = modelStep(this, params)
            % Forward pass, loss computation, and gradient calculation.
            %
            % Computes a weighted combination of L1 pixel loss and the
            % differentiable SSIM loss (1 - SSIM), following the original
            % 3DGS paper's loss formulation.

            % Forward pass: render current Gaussian parameters to this.image
            this.createImage(params);

            % L1 pixel loss — robust to outlier splat contributions
            l1_loss = mean(abs(this.image - this.image_gt), 'all');

            % Differentiable SSIM loss — penalises structural differences
            ssim_loss = 1.0 - this.ssimLoss();

            % Combined weighted loss: lambda balances pixel fidelity vs. structure
            loss = (1.0 - this.lambda) * l1_loss + this.lambda * ssim_loss;

            % Compute gradients with respect to all learnable parameters
            grads = dlgradient(loss, params);
        end

        function ssim_val = ssimLoss(this)
            % Compute the differentiable Structural Similarity (SSIM) metric.
            %
            % Uses SEPARABLE depthwise dlconv with 1-D kernels. Instead of
            % applying a full 2-D kernel (121 coefficients per pixel),
            % apply horizontal then vertical 1-D kernels. This reduces
            % FLOPs and memory for temporaries by ~50%.
            %
            % The horizontal [1 × wS × C × C] and vertical [wS × 1 × C × C]
            % kernels have diagonal structure ensuring channel independence.

            % Separable Gaussian convolution: apply horizontal then vertical
            mu1_h = dlconv(this.image,    this.window_h, 0, 'Padding', 'same');
            mu1   = dlconv(mu1_h,         this.window_v, 0, 'Padding', 'same');

            mu2_h = dlconv(this.image_gt, this.window_h, 0, 'Padding', 'same');
            mu2   = dlconv(mu2_h,         this.window_v, 0, 'Padding', 'same');

            mu1_sq  = mu1.^2;
            mu2_sq  = mu2.^2;
            mu1_mu2 = mu1 .* mu2;

            % Variance terms using separable convolution
            im1_sq_h = dlconv(this.image.^2,               this.window_h, 0, 'Padding', 'same');
            sigma1_sq = dlconv(im1_sq_h,                   this.window_v, 0, 'Padding', 'same') - mu1_sq;

            im2_sq_h = dlconv(this.image_gt.^2,            this.window_h, 0, 'Padding', 'same');
            sigma2_sq = dlconv(im2_sq_h,                   this.window_v, 0, 'Padding', 'same') - mu2_sq;

            im12_h = dlconv(this.image .* this.image_gt,   this.window_h, 0, 'Padding', 'same');
            sigma12 = dlconv(im12_h,                       this.window_v, 0, 'Padding', 'same') - mu1_mu2;

            % SSIM map: numerator captures cross-correlation; denominator
            % normalises by the sum of individual variances.
            ssim_map = ((single(2.0) .* mu1_mu2 + this.C1) .* (single(2.0) .* sigma12   + this.C2)) ./ ...
                       ((mu1_sq + mu2_sq + this.C1)         .* (sigma1_sq   + sigma2_sq  + this.C2));

            % Scalar SSIM: average over all spatial locations and channels
            ssim_val = mean(ssim_map, 'all');
        end

        function createImage(this, params)
            % Tile-aware Gaussian Splatting rasterizer
            %
            % STAGE 1 (CPU): buildTileList() extracts bounding boxes ONCE
            % per batch, pre-computes which Gaussians overlap each tile.
            %
            % STAGE 2 (GPU): rasterizeToGPU() processes pre-built tile lists
            %   WITHOUT any extractdata() calls in the inner loop.
            %
            % The current block canvas (imageHeight x imageWidth) is
            % partitioned into blockSize tiles. Per-tile Gaussian culling
            % reduces work by skipping Gaussians whose bounding box does not
            % overlap the tile (canvas is usually only 1-2 tiles now that
            % blocks and tiles share the same blockSize).
            %
            % Gaussian falloff uses the standard 3DGS Mahalanobis distance:
            %   alpha_k(p) = exp(-0.5 * d^T * Sigma2D_inv_k * d)
            % where d = [px - u_k; py - v_k] is the pixel offset from the
            % Gaussian centre, and Sigma2D_inv_k is the 2x2 inverse screen
            % covariance computed in projectGaussiansWithCulling via J*Sigma_cam*J^T.
            %
            % Pixel radii for tile culling use separate horizontal and vertical
            % bounds derived from eigenvalues of Sigma2D (3-sigma coverage).

            % Reset rendered image to zero for this mini-batch
            this.image(:) = 0;

            % Project all Gaussians; fills persistent buffers in-place
            % Returns the number of valid Gaussians for this batch
            numValid = this.projectGaussiansWithCulling(params);

            % Handle case where no valid Gaussians project to screen
            if numValid == 0
                return;
            end

            % STAGE 1 (CPU): Build tile lists once per batch
            % Extract bounding boxes once, pre-compute tile membership.
            % This is the only extractdata() call per batch in the rasterization flow.
            this.tileList = this.buildTileList(numValid);

            % STAGE 2 (GPU): Rasterize using pre-built tile lists
            % No extractdata() in the inner loop — all indexing is pre-computed
            this.rasterizeToGPU(numValid);
        end

        function tileList = buildTileList(this, numValid)
            % CPU-based tile culling: build Gaussian-to-tile membership lists,
            % one list per batch element (each element is a different block
            % with different per-block camera intrinsics).
            %
            % Fully vectorized: one GPU→CPU sync for the whole batch, then
            % per-tile membership via vectorized find() — no per-Gaussian
            % append loop. Also caches CPU bounding boxes (uminCPU etc.) so
            % rasterizeToGPU needs no per-Gaussian sync.

            blockH = this.blockSize(1);
            blockW = this.blockSize(2);

            numTilesH = ceil(this.imageHeight / blockH);
            numTilesW = ceil(this.imageWidth  / blockW);

            % Extract bounding boxes once for all batch elements (SINGLE SYNC)
            u_all  = gather(reshape(extractdata(this.u_buffer(1:numValid, 1, 1, :)), numValid, []));
            v_all  = gather(reshape(extractdata(this.v_buffer(1:numValid, 1, 1, :)), numValid, []));
            ru_all = gather(reshape(this.radii_u_buffer(1:numValid, 1, :), numValid, []));
            rv_all = gather(reshape(this.radii_v_buffer(1:numValid, 1, :), numValid, []));

            % CPU bounding boxes [numValid x B], reused by rasterizeToGPU.
            % Slots beyond an element's own valid count are zeroed upstream
            % → degenerate boxes → excluded by the overlap masks below.
            this.uminCPU = max(single(1.0), floor(u_all) - ru_all);
            this.umaxCPU = min(single(this.imageWidth),  ceil(u_all) + ru_all);
            this.vminCPU = max(single(1.0), floor(v_all) - rv_all);
            this.vmaxCPU = min(single(this.imageHeight), ceil(v_all) + rv_all);

            tileList = cell(1, this.miniBatchSize);

            for b = 1:this.miniBatchSize
                umin = this.uminCPU(:, b);
                umax = this.umaxCPU(:, b);
                vmin = this.vminCPU(:, b);
                vmax = this.vmaxCPU(:, b);

                nondegenerate = (umin <= umax) & (vmin <= vmax);

                tiles = cell(numTilesH, numTilesW);
                for th = 1:numTilesH
                    tVmin = single((th - 1) * blockH + 1);
                    tVmax = single(min(th * blockH, this.imageHeight));
                    for tw = 1:numTilesW
                        tUmin = single((tw - 1) * blockW + 1);
                        tUmax = single(min(tw * blockW, this.imageWidth));

                        % find() preserves ascending index order == depth order
                        tiles{th, tw} = find(nondegenerate & ...
                            umin <= tUmax & umax >= tUmin & ...
                            vmin <= tVmax & vmax >= tVmin);
                    end
                end

                tileList{b} = tiles;
            end
        end

        function rasterizeToGPU(this, numValid) %#ok<INUSD>
            % Chunked, vectorized alpha compositing per tile.
            %
            % Painter's algorithm reformulated without a serial per-Gaussian
            % dependency:  C(p) = sum_k c_k*alpha_k(p)*prod_{j<k}(1-alpha_j(p))
            %
            % Depth-sorted Gaussians are processed in chunks of chunkSize:
            %   1. Alpha maps for the whole chunk in one fused broadcast
            %      [H' x W' x K] — one set of kernel launches per chunk
            %      instead of per Gaussian.
            %   2. Within-chunk transmittance via cumprod along dim 3,
            %      detached from autodiff (extractdata) — same gradient
            %      semantics as the previous per-Gaussian version, which
            %      also excluded T from the tape.
            %   3. Tile output accumulated in local accR/accG/accB and
            %      written to this.image ONCE per tile, avoiding repeated
            %      dlarray copy-on-write subscripted assignments.

            blockH = this.blockSize(1);
            blockW = this.blockSize(2);
            K      = this.chunkSize;

            for b = 1:this.miniBatchSize
                % Tile lists for this batch element's block/camera view
                tiles = this.tileList{b};

                for th = 1:size(tiles, 1)
                    tileVmin = (th - 1) * blockH + 1;
                    tileVmax = min(th * blockH, this.imageHeight);

                    for tw = 1:size(tiles, 2)
                        gaussianList = tiles{th, tw};
                        if isempty(gaussianList)
                            continue;
                        end

                        tileUmin = (tw - 1) * blockW + 1;
                        tileUmax = min(tw * blockW, this.imageWidth);

                        x_t = this.X_vec(tileUmin:tileUmax);  % [1 x W']
                        y_t = this.Y_vec(tileVmin:tileVmax);  % [H' x 1]

                        % Tile-local transmittance carried across chunks
                        T_tile = ones(numel(y_t), numel(x_t), 'single', 'gpuArray');

                        accR = single(0);
                        accG = single(0);
                        accB = single(0);

                        numG = numel(gaussianList);
                        for c0 = 1:K:numG
                            idx = gaussianList(c0:min(c0 + K - 1, numG));

                            % Chunk parameters as [1 x 1 x K] for broadcasting
                            u_c = reshape(stripdims(this.u_buffer(idx, 1, 1, b)), 1, 1, []);
                            v_c = reshape(stripdims(this.v_buffer(idx, 1, 1, b)), 1, 1, []);
                            a_c = reshape(stripdims(this.alphas_buffer(idx, 1, 1, b)), 1, 1, []);
                            s11 = reshape(stripdims(this.Sigma2D_inv_buffer(idx, 1, 1, b)), 1, 1, []);
                            s12 = reshape(stripdims(this.Sigma2D_inv_buffer(idx, 1, 2, b)), 1, 1, []);
                            s22 = reshape(stripdims(this.Sigma2D_inv_buffer(idx, 2, 2, b)), 1, 1, []);

                            dx = x_t - u_c;  % [1  x W' x K]
                            dy = y_t - v_c;  % [H' x 1  x K]

                            % Fused Mahalanobis falloff for the whole chunk
                            alpha = a_c .* exp(single(-0.5) .* ...
                                (s11 .* dx.^2 + single(2.0) .* s12 .* (dy .* dx) + ...
                                 s22 .* dy.^2));  % [H' x W' x K]

                            % Transmittance is detached from autodiff (as before);
                            % Tprev(:,:,k) = T before compositing chunk-Gaussian k
                            alpha_nd = extractdata(alpha);
                            Tk    = cumprod(single(1.0) - alpha_nd, 3);
                            Tprev = cat(3, T_tile, T_tile .* Tk(:, :, 1:end-1));

                            w = alpha .* Tprev;  % dlarray composite weights

                            cR = reshape(stripdims(this.colors_buffer(idx, 1, 1, b)), 1, 1, []);
                            cG = reshape(stripdims(this.colors_buffer(idx, 1, 2, b)), 1, 1, []);
                            cB = reshape(stripdims(this.colors_buffer(idx, 1, 3, b)), 1, 1, []);

                            accR = accR + sum(w .* cR, 3);
                            accG = accG + sum(w .* cG, 3);
                            accB = accB + sum(w .* cB, 3);

                            T_tile = T_tile .* Tk(:, :, end);
                        end

                        % Single write per tile (tiles are disjoint, image pre-zeroed)
                        this.image(tileVmin:tileVmax, tileUmin:tileUmax, :, b) = ...
                            cat(3, accR, accG, accB);
                    end
                end
            end
        end

        function numValid = projectGaussiansWithCulling(this, params)
            % Project 3D Gaussians to screen space with frustum culling.
            %
            % All projections are in block-local pixel coordinates using the
            % per-block camera intrinsics (cx, cy already shifted by
            % ColmapData to this block's origin). The rasterizer clips to
            % tiles after projection, not before. This ensures Gaussians that
            % span multiple tiles are handled correctly.
            %
            % Implements the standard Kerbl et al. 2023 covariance projection:
            %
            %   1. Build 3D world covariance:
            %        Sigma_world = R * diag(s)^2 * R^T
            %      where R is the 3x3 rotation matrix from the unit quaternion
            %      and s = exp(scales_raw) are the positive semi-axis lengths.
            %
            %   2. Transform to camera space:
            %        Sigma_cam = Rcw * Sigma_world * Rcw^T
            %
            %   3. Project to 2D screen space via affine Jacobian:
            %        J = [fx/z,  0,    -fx*x/z^2]
            %            [0,     fy/z, -fy*y/z^2]
            %        Sigma_2D = J * Sigma_cam * J^T
            %
            %   4. Low-pass filter (prevents sub-pixel singularities):
            %        Sigma_2D += 0.3 * I_2
            %
            %   5. Invert 2x2 analytically:
            %        Sigma_2D_inv = inv(Sigma_2D)
            %
            %   6. Pixel radii for tile culling (separate H and V, 3-sigma):
            %        lambda_u = largest eigenvalue (used for horizontal radius)
            %        lambda_v = smallest eigenvalue (used for vertical radius)
            %        radius_u = ceil(3 * sqrt(lambda_u))
            %        radius_v = ceil(3 * sqrt(lambda_v))
            %
            % Returns:
            %   numValid: number of Gaussians that passed frustum culling
            %
            % OUTPUT BUFFERS (persistent, reused every batch):
            %   this.u_buffer           [numGaussians x 1 x 1 x B]
            %   this.v_buffer           [numGaussians x 1 x 1 x B]
            %   this.alphas_buffer      [numGaussians x 1 x 1 x B]
            %   this.Sigma2D_inv_buffer [numGaussians x 2 x 2 x B]
            %   this.radii_u_buffer     [numGaussians x 1 x B]
            %   this.radii_v_buffer     [numGaussians x 1 x B]
            %   this.colors_buffer      [numGaussians x 1 x 3 x B]

            % Zero persistent buffers at start of batch
            this.u_buffer(:) = 0;
            this.v_buffer(:) = 0;
            this.alphas_buffer(:) = 0;
            this.Sigma2D_inv_buffer(:) = 0;
            this.radii_u_buffer(:) = 0;
            this.radii_v_buffer(:) = 0;
            this.colors_buffer(:) = 0;

            % Transform world-space positions to camera space
            gaussians = pagemtimes( ...
                repmat(params.pws, 1, 1, this.miniBatchSize), 'none', ...
                stripdims(this.camera.Rcw), 'transpose');
            gaussians = gaussians + repmat( ...
                pagetranspose(reshape(stripdims(this.camera.tcw), 3, 1, this.miniBatchSize)), ...
                size(params.pws, 1), 1, 1);

            % Perspective projection: divide X,Y by Z, then apply full-frame intrinsics.
            % cx, cy, fx, fy correspond to the full downscaled image (not a tile).
            inv_z = single(1.0) ./ max(gaussians(:, 3, :), single(1e-6));
            gaussians(:, 1, :) = gaussians(:, 1, :) .* inv_z ...
                             .* reshape(this.camera.fx, 1, 1, this.miniBatchSize) ...
                             + reshape(this.camera.cx, 1, 1, this.miniBatchSize);
            gaussians(:, 2, :) = gaussians(:, 2, :) .* inv_z ...
                             .* reshape(this.camera.fy, 1, 1, this.miniBatchSize) ...
                             + reshape(this.camera.cy, 1, 1, this.miniBatchSize);

            % Frustum culling: keep Gaussians within depth bounds and a
            % screen-margin safety window. The margin has a 128 px floor so
            % large splats reaching into a small block canvas from outside
            % are not culled (50% of a ~100 px block would be too tight).
            marginW = max(single(this.imageWidth)  * single(0.5), single(128));
            marginH = max(single(this.imageHeight) * single(0.5), single(128));
            valid = (gaussians(:, 3, :) >  single(0.2))                            & ...
                    (gaussians(:, 3, :) <  single(100.0))                          & ...
                    (gaussians(:, 1, :) > -marginW)                                & ...
                    (gaussians(:, 1, :) <  single(this.imageWidth)  + marginW)     & ...
                    (gaussians(:, 2, :) > -marginH)                                & ...
                    (gaussians(:, 2, :) <  single(this.imageHeight) + marginH);

            % If no guassians splat in 2d image return empty image
            nr_valid = sum(valid, 1);
            if ~any(nr_valid)
                numValid = 0;
                return;
            end

            % Extract only once per batch
            for j = 1:this.miniBatchSize
                % Sort surviving Gaussians back-to-front by depth (Painter's Algorithm)
                valid_b = extractdata(valid(:, :, j));
                gaussians_z_b = extractdata(gaussians(valid_b, 3, j));

                [~, sortIdx] = sort(gaussians_z_b, 'descend');
                validIdx     = find(valid_b);
                sortIdx      = validIdx(sortIdx);
                numIdx       = length(sortIdx);

                if numIdx == 0
                    continue;
                end

                % Screen-space centres in block-local pixel coordinates
                this.u_buffer(1:numIdx, 1, 1, j) = gaussians(sortIdx, 1, j);
                this.v_buffer(1:numIdx, 1, 1, j) = gaussians(sortIdx, 2, j);

                % Camera-space positions of surviving Gaussians (before projection)
                % Needed for J computation: x_cam, y_cam, z_cam
                % IMPORTANT: Avoid recomputing camera-space positions.
                % The gaussians variable contains world→camera transformation,
                % but we need camera-space coords for the Jacobian before perspective proj.
                x_cam_full = pagemtimes( ...
                    repmat(params.pws, 1, 1, 1), 'none', ...
                    this.camera.Rcw(:, :, j), 'transpose') ...
                    + pagetranspose(reshape(this.camera.tcw(:, j), 3, 1, 1));
                x_cam = x_cam_full(sortIdx, 1);
                y_cam = x_cam_full(sortIdx, 2);
                z_cam = max(x_cam_full(sortIdx, 3), single(1e-6));

                % Opacity: sigmoid activation of raw logit
                % Clamp to [0, 1] for numerical stability
                alphas_raw = single(1.0) ./ (single(1.0) + exp(-params.alphas_raw(sortIdx)));
                this.alphas_buffer(1:numIdx, 1, 1, j) = min(alphas_raw, single(1.0));

                % Normalize quaternions to unit length before rotation conversion
                quat = params.rots_raw(sortIdx, :);
                quat = quat ./ max(vecnorm(quat, 2, 2), 1e-6);

                % Vectorized quaternion → rotation matrix construction
                qw = quat(:, 1); qx = quat(:, 2); qy = quat(:, 3); qz = quat(:, 4);

                % Build all 9 columns of rotation matrix at once
                R_cols = [single(1.0) - single(2.0) .* (qy .* qy + qz .* qz), ...
                          single(2.0) .* (qx .* qy - qw .* qz), ...
                          single(2.0) .* (qx .* qz + qw .* qy), ...
                          single(2.0) .* (qx .* qy + qw .* qz), ...
                          single(1.0) - single(2.0) .* (qx .* qx + qz .* qz), ...
                          single(2.0) .* (qy .* qz - qw .* qx), ...
                          single(2.0) .* (qx .* qz - qw .* qy), ...
                          single(2.0) .* (qy .* qz + qw .* qx), ...
                          single(1.0) - single(2.0) .* (qx .* qx + qy .* qy)];

                Sigma = reshape(R_cols', 3, 3, numIdx);  % Reshape [numIdx×9] → [3×3×numIdx]

                % Build 3D world covariance Sigma_world = Sigma*S^2*Sigma^T
                sx = exp(min(max(params.scales_raw(sortIdx, 1), single(-10)), single(10)));
                sy = exp(min(max(params.scales_raw(sortIdx, 2), single(-10)), single(10)));
                sz = exp(min(max(params.scales_raw(sortIdx, 3), single(-10)), single(10)));

                % Sigma = Sigma * diag(s): scale each column of Sigma in-place
                Sigma(1, 1, :) = Sigma(1, 1, :) .* reshape(sx, 1, 1, []);
                Sigma(2, 1, :) = Sigma(2, 1, :) .* reshape(sx, 1, 1, []);
                Sigma(3, 1, :) = Sigma(3, 1, :) .* reshape(sx, 1, 1, []);
                Sigma(1, 2, :) = Sigma(1, 2, :) .* reshape(sy, 1, 1, []);
                Sigma(2, 2, :) = Sigma(2, 2, :) .* reshape(sy, 1, 1, []);
                Sigma(3, 2, :) = Sigma(3, 2, :) .* reshape(sy, 1, 1, []);
                Sigma(1, 3, :) = Sigma(1, 3, :) .* reshape(sz, 1, 1, []);
                Sigma(2, 3, :) = Sigma(2, 3, :) .* reshape(sz, 1, 1, []);
                Sigma(3, 3, :) = Sigma(3, 3, :) .* reshape(sz, 1, 1, []);

                % Sigma_world = Sigma * Sigma^T
                Sigma = pagemtimes(Sigma, 'none', Sigma, 'transpose');

                % Transform to camera space
                Rcw_b = this.camera.Rcw(:, :, j);
                Sigma = pagemtimes(pagemtimes(Rcw_b, Sigma), pagetranspose(Rcw_b));

                % Project to 2D using affine Jacobian J (2x3)
                fx  = single(this.camera.fx(j));
                fy  = single(this.camera.fy(j));
                iz  = reshape(single(1.0) ./ z_cam, 1, 1, []);
                iz2 = iz .* iz;
                xc  = reshape(x_cam, 1, 1, []);
                yc  = reshape(y_cam, 1, 1, []);

                J          = dlarray(gpuArray(zeros(2, 3, numIdx, 'single')));
                J(1, 1, :) = fx .* iz;
                J(1, 2, :) = single(0.0);
                J(1, 3, :) = -fx .* xc .* iz2;
                J(2, 1, :) = single(0.0);
                J(2, 2, :) = fy .* iz;
                J(2, 3, :) = -fy .* yc .* iz2;

                % Sigma_2D = J * Sigma_cam * J^T
                Sigma = pagemtimes(pagemtimes(J, Sigma), pagetranspose(J));

                % Low-pass filter — add 0.3*I to prevent singularities
                % when a Gaussian projects to sub-pixel size.
                Sigma(1, 1, :) = Sigma(1, 1, :) + single(0.3);
                Sigma(2, 2, :) = Sigma(2, 2, :) + single(0.3);

                % Invert 2x2 covariance analytically.
                a   = Sigma(1, 1, :);
                b   = Sigma(1, 2, :);
                d   = Sigma(2, 2, :);
                inv_det = single(1.0) ./ max(a .* d - b .* b, single(1e-6));

                a_r = reshape(a, [], 1);
                b_r = reshape(b, [], 1);
                d_r = reshape(d, [], 1);
                i_r = reshape(inv_det, [], 1);
                
                this.Sigma2D_inv_buffer(1:numIdx, 1, 1, j) =  d_r .* i_r;
                this.Sigma2D_inv_buffer(1:numIdx, 1, 2, j) = -b_r .* i_r;
                this.Sigma2D_inv_buffer(1:numIdx, 2, 1, j) = -b_r .* i_r;
                this.Sigma2D_inv_buffer(1:numIdx, 2, 2, j) =  a_r .* i_r;

                % Separate horizontal and vertical pixel radii for tile culling
                a_data = squeeze(extractdata(a));
                d_data = squeeze(extractdata(d));
                b_data = squeeze(extractdata(b));

                mid   = single(0.5) .* (a_data + d_data);
                delta = sqrt(max(single(0.25) .* (a_data - d_data).^2 + b_data.^2, single(0.0)));
                lam_max  = mid + delta; % larger eigenvalue

                this.radii_u_buffer(1:numIdx, 1, j) = ceil(single(3.0) .* sqrt(max(lam_max, single(0.0))));
                this.radii_v_buffer(1:numIdx, 1, j) = ceil(single(3.0) .* sqrt(max(lam_max, single(0.0))));

                % Get world-space camera center
                C_world = this.camera.twc(:, j);  % Camera center in world coordinates

                % Get world-space Gaussian positions (use original params.pws, NOT gaussians)
                view_dir = params.pws(sortIdx, :);  % [numIdx × 3]

                % Compute view direction: from camera center to Gaussian
                view_dir = view_dir - repmat(C_world', numIdx, 1);  % [numIdx × 3]

                % Unit-normalise direction for spherical harmonic evaluation
                view_dir = view_dir ./ max(vecnorm(view_dir, 2, 2), 1e-6);

                % Evaluate second-order spherical harmonics for RGB color
                Sh = zeros(numIdx, length(this.shToColor), 'single');
                Sh(:, 1) = this.shToColor(1);
                Sh(:, 2) = this.shToColor(2) .* (-view_dir(:, 1));
                Sh(:, 3) = this.shToColor(3) .* (-view_dir(:, 2));
                Sh(:, 4) = this.shToColor(4) .* ( view_dir(:, 3));
                Sh(:, 5) = this.shToColor(5) .* ( view_dir(:, 1) .* view_dir(:, 2));
                Sh(:, 6) = this.shToColor(6) .* (-view_dir(:, 1) .* view_dir(:, 3));
                Sh(:, 7) = this.shToColor(7) .* (-view_dir(:, 2) .* view_dir(:, 3));
                Sh(:, 8) = this.shToColor(8) .* (single(3.0) .* view_dir(:, 3) .* view_dir(:, 3) - single(1.0));
                Sh(:, 9) = this.shToColor(9) .* (view_dir(:, 1) .* view_dir(:, 1) - view_dir(:, 2) .* view_dir(:, 2));

                % Clamp colors to valid [0, 1] range after SH expansion
                this.colors_buffer(1:numIdx, 1, 1, j) = dlarray(max(min( ...
                    single(0.5) + sum(Sh .* params.shs(sortIdx, :, 1), 2), single(1.0)), single(0.0)), 'SSC');
                this.colors_buffer(1:numIdx, 1, 2, j) = dlarray(max(min( ...
                    single(0.5) + sum(Sh .* params.shs(sortIdx, :, 2), 2), single(1.0)), single(0.0)), 'SSC');
                this.colors_buffer(1:numIdx, 1, 3, j) = dlarray(max(min( ...
                    single(0.5) + sum(Sh .* params.shs(sortIdx, :, 3), 2), single(1.0)), single(0.0)), 'SSC');
            end

            numValid = max(extractdata(nr_valid));
        end

        function pruneAndDensify(this, avgGrad, avgSqGrad, prunningRatio)
            % Adaptive densification: prune low-contribution Gaussians and
            % replace them with clones or splits of high-gradient Gaussians.
            %
            % Pruning strategy:
            %   - Low opacity x low gradient -> dead weight -> prune.
            %   - Force-prune if opacity logit < forcePruneThreshold (~0.018).
            %   - Force-keep if opacity logit > forceKeepThreshold  (~0.88).
            %
            % Densification strategy:
            %   - Clone: duplicate a high-gradient Gaussian at the same location.
            %   - Split: reduce scale by splitScaleFactor on both halves.


            % Sort by combined opacity-gradient score (low score → prune first)
            [~, pruneIdx] = sort(this.params.alphas_raw .* abs(avgGrad.alphas_raw));

            % Candidates: below force-prune threshold
            shouldPrune = (this.params.alphas_raw(pruneIdx) < this.forcePruneThreshold);

            % Also mark the lowest-scoring prunningRatio fraction
            shouldPrune(1:ceil(prunningRatio * this.numGaussians)) = true;

            % Exempt Gaussians with opacity above the keep threshold
            shouldPrune = shouldPrune & (this.params.alphas_raw(pruneIdx) < this.forceKeepThreshold);

            numPrune = sum(shouldPrune);
            if numPrune
                pruneIdx = pruneIdx(shouldPrune);

                % Select densification candidates: highest positional gradient
                [~, cloneIdx] = sort(vecnorm(avgGrad.pws, 2, 2), 'descend');

                % Remove any index already scheduled for pruning
                cloneIdx = setdiff(extractdata(cloneIdx), extractdata(pruneIdx), 'stable');
                cloneIdx = cloneIdx(1:numPrune);

                % Decide split vs. clone based on current scale
                shouldSplit = max(exp(this.params.scales_raw), [], 2) > ...
                              this.scaleThresh * this.sceneScale;

                % Replace pruned slots with clones (or splits)
                this.params.pws(pruneIdx, :)        = this.params.pws(cloneIdx, :);
                this.params.alphas_raw(pruneIdx, :) = this.params.alphas_raw(cloneIdx, :);
                this.params.scales_raw(pruneIdx, :) = this.params.scales_raw(cloneIdx, :) ...
                                                     - shouldSplit(cloneIdx, :) .* this.splitScaleFactor;
                this.params.scales_raw(cloneIdx, :) = this.params.scales_raw(cloneIdx, :) ...
                                                     - shouldSplit(cloneIdx, :) .* this.splitScaleFactor;
                this.params.rots_raw(pruneIdx, :)   = this.params.rots_raw(cloneIdx, :);
                this.params.shs(pruneIdx, :)        = this.params.shs(cloneIdx, :);

                % Reset all fields to prevent stale momentum from affecting new Gaussians
                avgGrad.pws(pruneIdx, :)        = 0;
                avgGrad.shs(pruneIdx, :, :)     = 0;
                avgGrad.scales_raw(pruneIdx, :) = 0;
                avgGrad.alphas_raw(pruneIdx, :) = 0;
                avgGrad.rots_raw(pruneIdx, :)   = 0;

                avgSqGrad.pws(pruneIdx, :)        = 0;
                avgSqGrad.shs(pruneIdx, :, :)     = 0;
                avgSqGrad.scales_raw(pruneIdx, :) = 0;
                avgSqGrad.alphas_raw(pruneIdx, :) = 0;
                avgSqGrad.rots_raw(pruneIdx, :)   = 0;

                fprintf('Densification: %d clones, %d splits.\n', ...
                    sum(~shouldSplit(cloneIdx, :)), sum(shouldSplit(cloneIdx, :)));
            end
        end

        function saveGaussians(this, filename)
            % Gather GPU arrays back to CPU and save learnable parameters.
            params = gather(this.params); %#ok<PROPLC>
            save(filename, "params");
        end

        function printGPUMemory(this, label)
            % Lightweight GPU memory diagnostics.
            % Call after initStorage, densification, or key training points
            % to verify memory usage is stable and not fragmenting.
            g = gpuDevice;
            used = g.TotalMemory - g.AvailableMemory;
            fprintf('[%s] GPU memory used: %.2f MB / %.2f MB\n', ...
                label, used/1e6, g.TotalMemory/1e6);
        end
    end

    methods (Static, Access = private)
        function paramStruct = createLearnableParams(gaussians)
            % Convert Gaussian struct of doubles to a struct of dlarrays.
            % All parameters start on CPU; initStorage migrates them to GPU.

            paramStruct.pws = dlarray(single(gaussians.pws));

            % Reshape SH coefficients to [N × 9 × 3]
            paramStruct.shs = dlarray(single( ...
                resize(reshape(gaussians.shs, [], 1, 3), 9, dimension=2)));

            % Scale: store in log domain so exp() gives positive scale
            paramStruct.scales_raw = dlarray(single(log(gaussians.scales + 1e-6)));

            % Opacity: store as logit (inverse sigmoid) for unconstrained optimisation
            a = gaussians.alphas;
            a = max(min(a, 0.99), 0.01);
            paramStruct.alphas_raw = dlarray(single(log(a ./ (1 - a))));

            % Rotation quaternions: passed through unchanged
            paramStruct.rots_raw = dlarray(single(gaussians.rots));
        end

        function [window_h, window_v] = createSeparableWindow(window_size, channel)
            % Build separable 1-D Gaussian kernels for SSIM loss
            %
            % Instead of a full [wS × wS] 2-D kernel (121 coefficients per pixel),
            % create two [wS × 1] and [1 × wS] kernels with diagonal structure.
            % This enables two 1-D depthwise convolutions instead of one 2-D,
            % reducing FLOPs by ~50% and temporary memory similarly.
            %
            % A separable 1D Gaussian (sigma=1.5) is used for both kernels.
            % The diagonal [wS × wS × C × C] structure ensures each channel
            % is filtered independently.

            sigma    = single(1.5);
            coords   = 0:single(window_size - 1);
            center   = floor(window_size / 2);
            gauss    = exp(-(coords - center).^2 / (2 * sigma^2));
            gauss    = gauss / sum(gauss);  % Normalise to unit sum

            % Horizontal kernel: [1 x wS x C x C] with diagonal structure
            window_h = zeros(1, window_size, channel, channel, 'single');
            for c = 1:channel
                window_h(1, :, c, c) = gauss;
            end

            % Vertical kernel: [wS x 1 x C x C] with diagonal structure
            window_v = zeros(window_size, 1, channel, channel, 'single');
            for c = 1:channel
                window_v(:, 1, c, c) = gauss(:);
            end
        end
    end
end