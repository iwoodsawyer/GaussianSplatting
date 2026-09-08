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
    % Block-Aware Rasterization:
    %   this.data.images (ColmapData) now yields one blockedImageDatastore
    %   block per read(), sized blockSize + 2*overlap, with camera cx/cy
    %   already shifted to that block's local origin. Overlap between
    %   blocks is baked into ColmapData's overlapping BlockOffsets, so each
    %   block canvas is rasterized as a single unit — no further spatial
    %   tile subdivision happens here.
    %
    %   Gaussian culling is performed ONCE on CPU per batch (via
    %   buildTileList), then rasterizeToGPU processes the pre-built lists
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
    %   Projection buffers are allocated ONCE and fully overwritten each
    %   batch (no zeroing pass needed), eliminating fragmentation and
    %   per-batch allocation overhead. minibatchqueue delivers data with
    %   OutputEnvironment='gpu', so GPU residency is guaranteed from the
    %   first iteration onwards.
    %
    % GPU Optimization Strategy:
    %   - Projection fully vectorized across the batch: [N x B] arrays,
    %     elementwise J*Sigma*J^T, one column-wise depth sort, buffers
    %     filled by pure indexing (no per-batch-element compute loop)
    %   - Rasterization in depth-sorted chunks of chunkSize: fused
    %     [H' x W' x K] alpha maps, cumprod transmittance, tile-local
    %     accumulation with a single image write per tile
    %   - Minimal GPU→CPU syncs: culling/sort inputs, tile-list bboxes,
    %     and one amortized early-termination check per chunk
    %   - Persistent buffers: reused across iterations (no per-batch malloc)
    %   - Separable operations: 1-D coordinates and SSIM convolutions

    properties
        % --- Sizes ---
        numImages
        numGaussians
        miniBatchSize
        imageWidth
        imageHeight
        imageChannel

        % --- Block Settings ---
        % Block dimensions shared by the dataset's blockedImage grid
        % (ColmapData) and the rasterizer canvas — each read() delivers
        % one block of blockSize + 2*overlap, rasterized as a single unit.
        %
        % Default: auto-selected based on GPU compute capability.
        blockSize = [];  % Will be auto-selected in constructor if not provided

        % Gaussians composited per vectorized chunk in rasterizeToGPU.
        % Larger = fewer kernel launches but more [H'×W'×K] temp memory.
        chunkSize = 64;

        % --- Data ---
        datasetPath
        data

        % --- Device ---
        % Auto-detected once in the constructor; drives OutputEnvironment for
        % every minibatchqueue. All GPU/CPU buffer allocation elsewhere just
        % follows the device residency of the data delivered by those queues.
        useGPU

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
        % Allocated in initStorage() and never deallocated. Every call to
        % projectGaussiansWithCulling() fully overwrites them with depth-sorted
        % values (invalid slots marked by radius -1). This eliminates per-batch
        % GPU memory allocation and fragmentation.
        u_buffer           % [numGaussians x 1 x 1 x B] dlarray 'SSCB'
        v_buffer           % [numGaussians x 1 x 1 x B] dlarray 'SSCB'
        alphas_buffer      % [numGaussians x 1 x 1 x B] dlarray 'SSCB'
        Sigma2D_inv_buffer % [numGaussians x 2 x 2 x B] dlarray 'SSSB'
        radii_u_buffer     % [numGaussians x 1 x B] plain gpuArray
        radii_v_buffer     % [numGaussians x 1 x B] plain gpuArray
        colors_buffer      % [numGaussians x 1 x 3 x B] dlarray 'SSCB'

        % --- CPU-based Gaussian Lists (cached per batch) ---
        % Pre-computed valid-Gaussian list per batch element, built once per
        % batch on CPU (each batch element is a different block/camera view).
        % No spatial tile subdivision — overlap is baked into ColmapData's
        % overlapping block grid, so each canvas is rasterized as one unit.
        tileList           % cell array {1 x miniBatchSize} of Gaussian-index vectors

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
            %   blockSize    (1x2 int): Block [H W]. Defaults to auto-select.
            %                          Drives the dataset's blockedImage grid
            %                          (ColmapData) and the rasterizer canvas size.
            %   overlap      (1x2 int): Halo [H W] added around each block via
            %                          overlapping BlockOffsets (ColmapData).
            %                          Defaults to [0 0].
            %
            % BLOCKSIZE AUTO-SELECTION:
            %   If blockSize is empty or not provided, autoSelectBlockSize()
            %   picks the size (searched per dimension in [64, 160]) that
            %   minimizes total blockedImageDatastore zero-padding summed
            %   over both the Half and Full resolution levels.

            if nargin < 4 || isempty(blockSize)
                blockSize = GaussianSplatter.autoSelectBlockSize(datasetPath);
            end
            if nargin < 5 || isempty(overlap)
                overlap = [0, 0];
            end

            this.camera    = struct;
            this.blockSize = blockSize;

            % Falls back to false if no usable GPU or Parallel Computing Toolbox
            try
                this.useGPU = canUseGPU();
            catch
                this.useGPU = false;
            end
            if this.useGPU
                fprintf('GaussianSplatter running in GPU mode.\n');
            else
                fprintf('GaussianSplatter running in CPU mode.\n');
            end

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

            % Reference array already resident on the right device (GPU or
            % CPU) — this.image_gt was produced by a minibatchqueue configured
            % with this.useGPU, so every 'like' allocation below just follows it.
            refArr = extractdata(this.image_gt);

            % Rendered image buffer, sized to the current block canvas
            % (blockSize + 2*overlap).
            this.image = dlarray( ...
                zeros(this.imageHeight, this.imageWidth, 3, ...
                      this.miniBatchSize, 'like', refArr), 'SSCB');

            % Pixel coordinate vectors (separable) — saves ~8 MB vs. dense
            % meshgrid. Span the current block canvas so
            % projectGaussiansWithCulling maps correctly with the per-block
            % intrinsics (cx, cy already shifted to the block's local origin).
            % 
            % Instead of [H×W×1] grids, use [1×W] and [H×1] vectors with
            % implicit broadcasting during Mahalanobis computation.
            this.X_vec = cast(single(1:this.imageWidth), 'like', refArr);     % [1 x W]
            this.Y_vec = cast(single((1:this.imageHeight)'), 'like', refArr); % [H x 1]

            % Per-pixel transmittance accumulator — reset to 1 per batch
            % Plain array (not dlarray) since T is not differentiated
            this.T = ones(this.imageHeight, this.imageWidth, 1, 'like', refArr);

            % SSIM loss constants (plain array, not dlarray)
            this.C1 = cast(this.C1, 'like', refArr);
            this.C2 = cast(this.C2, 'like', refArr);

            % Combined loss weight
            this.lambda = cast(this.lambda, 'like', refArr);

            % Separable SSIM Gaussian kernels (plain array)
            this.window_h = cast(this.window_h, 'like', refArr);
            this.window_v = cast(this.window_v, 'like', refArr);

            % --- Allocate Persistent Projection Buffers ---
            % Allocated once and fully overwritten by every call to
            % projectGaussiansWithCulling(). This avoids per-batch GPU
            % allocation, which can cause fragmentation and sync overhead.
            fprintf('Allocating persistent projection buffers (%d Gaussians)...\n', ...
                this.numGaussians);

            this.u_buffer = dlarray(zeros(this.numGaussians, 1, 1, ...
                miniBatchSize, 'like', refArr), 'SSCB');
            this.v_buffer = dlarray(zeros(this.numGaussians, 1, 1, ...
                miniBatchSize, 'like', refArr), 'SSCB');
            this.alphas_buffer = dlarray(zeros(this.numGaussians, 1, 1, ...
                miniBatchSize, 'like', refArr), 'SSCB');
            this.Sigma2D_inv_buffer = dlarray(zeros(this.numGaussians, 2, 2, ...
                miniBatchSize, 'like', refArr), 'SSSB');
            this.radii_u_buffer = zeros(this.numGaussians, 1, ...
                miniBatchSize, 'like', refArr);
            this.radii_v_buffer = zeros(this.numGaussians, 1, ...
                miniBatchSize, 'like', refArr);
            this.colors_buffer = dlarray(zeros(this.numGaussians, 1, 3, ...
                miniBatchSize, 'like', refArr), 'SSCB');

            % --- Move learnable parameters to match data residency ---
            % No fresh array to hang 'like' off here (these already exist as
            % CPU dlarrays), so gate the move on the same useGPU decision.
            if this.useGPU
                fprintf('Moving learnable parameters to GPU...\n');
                this.params.pws        = gpuArray(this.params.pws);
                this.params.shs        = gpuArray(this.params.shs);
                this.params.scales_raw = gpuArray(this.params.scales_raw);
                this.params.alphas_raw = gpuArray(this.params.alphas_raw);
                this.params.rots_raw   = gpuArray(this.params.rots_raw);
            end
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
            % Gaussian Splatting rasterizer over the current block canvas.
            %
            % STAGE 1 (CPU): buildTileList() extracts bounding boxes ONCE
            % per batch, pre-computes which Gaussians are valid for this
            % batch's canvas (block overlap is already baked into
            % ColmapData's block grid, so no further subdivision is needed).
            %
            % STAGE 2 (GPU): rasterizeToGPU() processes the pre-built lists
            %   WITHOUT any extractdata() calls in the inner loop.
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
            % CPU-based culling: build the valid-Gaussian list per batch
            % element (each element is a different block with different
            % per-block camera intrinsics). No spatial tile subdivision —
            % block overlap already comes from ColmapData's block grid.
            %
            % Fully vectorized: one GPU→CPU sync for the whole batch, then
            % vectorized find() per batch element. Also caches CPU bounding
            % boxes (uminCPU etc.) so rasterizeToGPU needs no per-Gaussian sync.

            % Extract bounding boxes once for all batch elements (SINGLE SYNC)
            u_all  = gather(reshape(extractdata(this.u_buffer(1:numValid, 1, 1, :)), numValid, []));
            v_all  = gather(reshape(extractdata(this.v_buffer(1:numValid, 1, 1, :)), numValid, []));
            ru_all = gather(reshape(this.radii_u_buffer(1:numValid, 1, :), numValid, []));
            rv_all = gather(reshape(this.radii_v_buffer(1:numValid, 1, :), numValid, []));

            % CPU bounding boxes [numValid x B], reused by rasterizeToGPU.
            % Invalid slots carry radius -1 (sentinel from projection)
            % → degenerate boxes → excluded by the nondegenerate mask below.
            this.uminCPU = max(single(1.0), floor(u_all) - ru_all);
            this.umaxCPU = min(single(this.imageWidth),  ceil(u_all) + ru_all);
            this.vminCPU = max(single(1.0), floor(v_all) - rv_all);
            this.vmaxCPU = min(single(this.imageHeight), ceil(v_all) + rv_all);

            tileList = cell(1, this.miniBatchSize);

            for b = 1:this.miniBatchSize
                nondegenerate = (this.uminCPU(:, b) <= this.umaxCPU(:, b)) & ...
                                (this.vminCPU(:, b) <= this.vmaxCPU(:, b));

                % find() preserves ascending index order == depth order
                tileList{b} = find(nondegenerate);
            end
        end

        function rasterizeToGPU(this, numValid) %#ok<INUSD>
            % Chunked, vectorized alpha compositing over the full block canvas.
            %
            % Painter's algorithm reformulated without a serial per-Gaussian
            % dependency:  C(p) = sum_k c_k*alpha_k(p)*prod_{j<k}(1-alpha_j(p))
            %
            % Depth-sorted Gaussians are processed in chunks of chunkSize:
            %   1. Alpha maps for the whole chunk in one fused broadcast
            %      [H x W x K] — one set of kernel launches per chunk
            %      instead of per Gaussian.
            %   2. Within-chunk transmittance via cumprod along dim 3,
            %      detached from autodiff (extractdata) — same gradient
            %      semantics as the previous per-Gaussian version, which
            %      also excluded T from the tape.
            %   3. Output accumulated in local accR/accG/accB and written
            %      to this.image ONCE per batch element, avoiding repeated
            %      dlarray copy-on-write subscripted assignments.

            K = this.chunkSize;

            for b = 1:this.miniBatchSize
                gaussianList = this.tileList{b};
                if isempty(gaussianList)
                    continue;
                end

                x_t = this.X_vec;  % [1 x W]
                y_t = this.Y_vec;  % [H x 1]

                % Canvas-local transmittance carried across chunks
                T_tile = ones(this.imageHeight, this.imageWidth, 'like', this.T);

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

                    dx = x_t - u_c;  % [1 x W x K]
                    dy = y_t - v_c;  % [H x 1 x K]

                    % Fused Mahalanobis falloff for the whole chunk
                    alpha = a_c .* exp(single(-0.5) .* ...
                        (s11 .* dx.^2 + single(2.0) .* s12 .* (dy .* dx) + ...
                         s22 .* dy.^2));  % [H x W x K]

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

                    % Per-chunk early termination: one small sync per
                    % chunk (amortized over chunkSize Gaussians);
                    % skipped on the final chunk where it's useless.
                    if c0 + K <= numG && ...
                            gather(max(T_tile, [], 'all')) < single(0.01)
                        break;
                    end
                end

                % Single write per batch element (image pre-zeroed)
                this.image(:, :, :, b) = cat(3, accR, accG, accB);
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
            % FULLY VECTORIZED ACROSS THE BATCH: batch-independent quantities
            % (opacity sigmoid, quaternion→rotation, world covariance) are
            % computed once for all N Gaussians; batch-dependent quantities
            % (camera transform, projection, Sigma2D, colors) as [N x B]
            % arrays; Sigma_2D = J*Sigma_cam*J^T is expanded elementwise so
            % no per-element J matrices are built. Depth sorting is done
            % column-wise and results are scattered into the persistent
            % buffers with pure indexing — no per-batch-element compute loop.
            %
            % OUTPUT BUFFERS (persistent, reused every batch):
            %   this.u_buffer           [numGaussians x 1 x 1 x B]
            %   this.v_buffer           [numGaussians x 1 x 1 x B]
            %   this.alphas_buffer      [numGaussians x 1 x 1 x B]
            %   this.Sigma2D_inv_buffer [numGaussians x 2 x 2 x B]
            %   this.radii_u_buffer     [numGaussians x 1 x B] (-1 = invalid slot)
            %   this.radii_v_buffer     [numGaussians x 1 x B] (-1 = invalid slot)
            %   this.colors_buffer      [numGaussians x 1 x 3 x B]

            N = size(params.pws, 1);
            B = this.miniBatchSize;

            % ---- Camera-space positions for all Gaussians x batch [N x 3 x B]
            p_cam = pagemtimes( ...
                repmat(params.pws, 1, 1, B), 'none', ...
                stripdims(this.camera.Rcw), 'transpose');
            p_cam = p_cam + repmat( ...
                pagetranspose(reshape(stripdims(this.camera.tcw), 3, 1, B)), N, 1, 1);

            x_cam  = reshape(p_cam(:, 1, :), N, B);
            y_cam  = reshape(p_cam(:, 2, :), N, B);
            z_cam  = reshape(p_cam(:, 3, :), N, B);
            z_safe = max(z_cam, single(1e-6));

            % ---- Perspective projection to block-local pixel coords [N x B]
            fxr = reshape(this.camera.fx, 1, B);
            fyr = reshape(this.camera.fy, 1, B);
            u_all = x_cam ./ z_safe .* fxr + reshape(this.camera.cx, 1, B);
            v_all = y_cam ./ z_safe .* fyr + reshape(this.camera.cy, 1, B);

            % ---- Frustum culling (plain gpuArray; culling needs no gradients).
            % Margin has a 128 px floor so large splats reaching into a small
            % block canvas from outside are not culled.
            marginW = max(single(this.imageWidth)  * single(0.5), single(128));
            marginH = max(single(this.imageHeight) * single(0.5), single(128));
            u_nd = extractdata(u_all);
            v_nd = extractdata(v_all);
            z_nd = extractdata(z_cam);
            valid = z_nd > single(0.2) & z_nd < single(100.0) & ...
                    u_nd > -marginW & u_nd < single(this.imageWidth)  + marginW & ...
                    v_nd > -marginH & v_nd < single(this.imageHeight) + marginH;

            nr_valid = sum(valid, 1);            % [1 x B]
            numValid = gather(max(nr_valid));
            if numValid == 0
                return;
            end

            % ---- Depth sort per batch column (back-to-front, invalid sink last)
            z_masked         = z_nd;
            z_masked(~valid) = -Inf;
            [~, sortOrder]   = sort(z_masked, 1, 'descend');  % [N x B]
            linIdx  = sortOrder + (0:B-1) * N;                % linear into [N x B]
            rowMask = (1:N)' <= nr_valid;                     % valid-slot mask

            % ---- Batch-independent quantities (computed ONCE, not per element)
            % Opacity: sigmoid of raw logit, clamped for stability
            alph = min(single(1.0) ./ (single(1.0) + exp(-params.alphas_raw)), single(1.0));

            % Quaternion → rotation matrix, vectorized over all N
            quat = params.rots_raw ./ max(vecnorm(params.rots_raw, 2, 2), 1e-6);
            qw = quat(:, 1); qx = quat(:, 2); qy = quat(:, 3); qz = quat(:, 4);

            R_cols = [single(1.0) - single(2.0) .* (qy .* qy + qz .* qz), ...
                      single(2.0) .* (qx .* qy - qw .* qz), ...
                      single(2.0) .* (qx .* qz + qw .* qy), ...
                      single(2.0) .* (qx .* qy + qw .* qz), ...
                      single(1.0) - single(2.0) .* (qx .* qx + qz .* qz), ...
                      single(2.0) .* (qy .* qz - qw .* qx), ...
                      single(2.0) .* (qx .* qz - qw .* qy), ...
                      single(2.0) .* (qy .* qz + qw .* qx), ...
                      single(1.0) - single(2.0) .* (qx .* qx + qy .* qy)];

            M = reshape(R_cols', 3, 3, N);

            % M = R * diag(s): scale each column by clamped exp(scales)
            sx = exp(min(max(params.scales_raw(:, 1), single(-10)), single(10)));
            sy = exp(min(max(params.scales_raw(:, 2), single(-10)), single(10)));
            sz = exp(min(max(params.scales_raw(:, 3), single(-10)), single(10)));

            M(1, 1, :) = M(1, 1, :) .* reshape(sx, 1, 1, []);
            M(2, 1, :) = M(2, 1, :) .* reshape(sx, 1, 1, []);
            M(3, 1, :) = M(3, 1, :) .* reshape(sx, 1, 1, []);
            M(1, 2, :) = M(1, 2, :) .* reshape(sy, 1, 1, []);
            M(2, 2, :) = M(2, 2, :) .* reshape(sy, 1, 1, []);
            M(3, 2, :) = M(3, 2, :) .* reshape(sy, 1, 1, []);
            M(1, 3, :) = M(1, 3, :) .* reshape(sz, 1, 1, []);
            M(2, 3, :) = M(2, 3, :) .* reshape(sz, 1, 1, []);
            M(3, 3, :) = M(3, 3, :) .* reshape(sz, 1, 1, []);

            % Sigma_world = M * M^T  [3 x 3 x N]
            Sw = pagemtimes(M, 'none', M, 'transpose');

            % ---- Camera-space covariance for all N x B (explicit page expansion)
            RcwN = repmat(reshape(stripdims(this.camera.Rcw), 3, 3, 1, B), 1, 1, N, 1);
            SwB  = repmat(reshape(Sw, 3, 3, N, 1), 1, 1, 1, B);
            Scam = pagemtimes(pagemtimes(RcwN, SwB), 'none', RcwN, 'transpose');

            S11 = reshape(Scam(1, 1, :, :), N, B);
            S12 = reshape(Scam(1, 2, :, :), N, B);
            S13 = reshape(Scam(1, 3, :, :), N, B);
            S22 = reshape(Scam(2, 2, :, :), N, B);
            S23 = reshape(Scam(2, 3, :, :), N, B);
            S33 = reshape(Scam(3, 3, :, :), N, B);

            % ---- Sigma_2D = J*Sigma_cam*J^T expanded elementwise.
            % J rows: j1 = [fx/z, 0, -fx*x/z^2], j2 = [0, fy/z, -fy*y/z^2],
            % so no [2x3xNxB] J matrices are materialized.
            t1 = fxr ./ z_safe;
            t3 = -fxr .* x_cam ./ (z_safe .^ 2);
            w2 = fyr ./ z_safe;
            w3 = -fyr .* y_cam ./ (z_safe .^ 2);

            % 0.3*I low-pass filter folded into the diagonal terms
            a  = t1.^2 .* S11 + single(2.0) .* t1 .* t3 .* S13 + t3.^2 .* S33 + single(0.3);
            bb = (t1 .* S12 + t3 .* S23) .* w2 + (t1 .* S13 + t3 .* S33) .* w3;
            d  = w2.^2 .* S22 + single(2.0) .* w2 .* w3 .* S23 + w3.^2 .* S33 + single(0.3);

            % Analytic 2x2 inverse [N x B]
            inv_det = single(1.0) ./ max(a .* d - bb .* bb, single(1e-6));
            i11 =  d  .* inv_det;
            i12 = -bb .* inv_det;
            i22 =  a  .* inv_det;

            % ---- Pixel radii from the larger Sigma2D eigenvalue (3-sigma, plain)
            a_nd = extractdata(a);
            b_nd = extractdata(bb);
            d_nd = extractdata(d);
            mid   = single(0.5) .* (a_nd + d_nd);
            delta = sqrt(max(single(0.25) .* (a_nd - d_nd).^2 + b_nd.^2, single(0.0)));
            r_all = ceil(single(3.0) .* sqrt(max(mid + delta, single(0.0))));  % [N x B]

            % ---- Spherical harmonic colors for all N x B
            tw_r = reshape(stripdims(this.camera.twc), 1, 3, B);
            vd   = params.pws - tw_r;                       % [N x 3 x B]
            vd   = vd ./ max(vecnorm(vd, 2, 2), 1e-6);
            vx   = vd(:, 1, :); vy = vd(:, 2, :); vz = vd(:, 3, :);

            c  = this.shToColor;
            Sh = cat(2, ...
                c(1) .* ones(N, 1, B, 'like', params.pws), ...
                c(2) .* (-vx), ...
                c(3) .* (-vy), ...
                c(4) .* vz, ...
                c(5) .* (vx .* vy), ...
                c(6) .* (-vx .* vz), ...
                c(7) .* (-vy .* vz), ...
                c(8) .* (single(3.0) .* vz .* vz - single(1.0)), ...
                c(9) .* (vx .* vx - vy .* vy));             % [N x 9 x B]

            colR = reshape(max(min(single(0.5) + sum(Sh .* params.shs(:, :, 1), 2), ...
                single(1.0)), single(0.0)), N, B);
            colG = reshape(max(min(single(0.5) + sum(Sh .* params.shs(:, :, 2), 2), ...
                single(1.0)), single(0.0)), N, B);
            colB = reshape(max(min(single(0.5) + sum(Sh .* params.shs(:, :, 3), 2), ...
                single(1.0)), single(0.0)), N, B);

            % ---- Scatter depth-sorted values into persistent buffers (indexing only)
            this.u_buffer(:, :, :, :)      = reshape(u_all(linIdx),   N, 1, 1, B);
            this.v_buffer(:, :, :, :)      = reshape(v_all(linIdx),   N, 1, 1, B);
            this.alphas_buffer(:, :, :, :) = reshape(alph(sortOrder), N, 1, 1, B);

            this.Sigma2D_inv_buffer(:, 1, 1, :) = reshape(i11(linIdx), N, 1, 1, B);
            this.Sigma2D_inv_buffer(:, 1, 2, :) = reshape(i12(linIdx), N, 1, 1, B);
            this.Sigma2D_inv_buffer(:, 2, 1, :) = reshape(i12(linIdx), N, 1, 1, B);
            this.Sigma2D_inv_buffer(:, 2, 2, :) = reshape(i22(linIdx), N, 1, 1, B);

            this.colors_buffer(:, 1, 1, :) = reshape(colR(linIdx), N, 1, 1, B);
            this.colors_buffer(:, 1, 2, :) = reshape(colG(linIdx), N, 1, 1, B);
            this.colors_buffer(:, 1, 3, :) = reshape(colB(linIdx), N, 1, 1, B);

            % Invalid slots get radius -1 → degenerate bbox → excluded downstream
            r_sorted = r_all(linIdx);
            r_sorted(~rowMask) = single(-1.0);
            this.radii_u_buffer(:, :, :) = reshape(r_sorted, N, 1, B);
            this.radii_v_buffer(:, :, :) = reshape(r_sorted, N, 1, B);
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

        function previewResults(this, gaussianParams, numGenImages)
            % Render numGenImages full images, each stitched back together
            % from its blocks, comparing prediction vs ground truth — rather
            % than showing individual (scrambled, out-of-order) blocks.
            blockH = this.data.blockSize(1);
            blockW = this.data.blockSize(2);
            % Overlap is trailing-only (baked into ColmapData's overlapping
            % block grid), so each block's core starts at its own origin.
            coreRows = 1:blockH;
            coreCols = 1:blockW;

            % Cheap metadata-only pass (no pixel decode) to group blocks by source image.
            camRows = readall(this.data.cameras);
            allIds  = cell2mat(camRows(:, 1));
            uniqueIds = unique(allIds, 'stable');
            numShow   = min(numGenImages, numel(uniqueIds));

            previewFormat = ["SSCB","CB","CB","CB","CB","CB","CB","CB","CB","CB","CB","SSCB","SCB","SCB"];

            for n = 1:numShow
                % All blocks belonging to this source image (blockId doubles as the
                % 1-based read-order index, so it can be used directly with subset()).
                blockIdx = find(allIds == uniqueIds(n));
                numBlocksForImage = numel(blockIdx);

                subDs  = subset(combine(this.data.images, this.data.cameras), blockIdx);
                subMbq = minibatchqueue(subDs, ...
                    'MiniBatchSize',    numBlocksForImage, ...
                    'OutputEnvironment',GaussianSplatter.outEnv(this.useGPU), ...
                    'MiniBatchFormat',  previewFormat);

                [this.image_gt, ~, ~, camBlockRow, camBlockCol, camW, camH, camFx, camFy, camCx, camCy, ...
                 this.camera.Rcw, this.camera.tcw, this.camera.twc] = next(subMbq);

                this.camera.width  = squeeze(camW);
                this.camera.height = squeeze(camH);
                this.camera.fx     = squeeze(camFx);
                this.camera.fy     = squeeze(camFy);
                this.camera.cx     = squeeze(camCx);
                this.camera.cy     = squeeze(camCy);

                this.initStorage(numBlocksForImage);
                this.createImage(gaussianParams);

                blockRows = round(gather(extractdata(squeeze(camBlockRow))));
                blockCols = round(gather(extractdata(squeeze(camBlockCol))));

                predBlocks = gather(extractdata(this.image));
                gtBlocks   = gather(extractdata(this.image_gt));

                % Reassemble via a writable blockedImage: setBlock() places each
                % core block at its tile-grid subscript, and gather() crops the
                % result to imgSz automatically (no manual stitching/edge-crop).
                imgNum = this.data.images.BlockLocationSet.ImageNumber(blockIdx(1));
                imgSz  = this.data.images.Images(imgNum).Size;

                bimPred = blockedImage([], imgSz, [blockH, blockW, 3], single(0), Mode="w");
                bimGT   = blockedImage([], imgSz, [blockH, blockW, 3], single(0), Mode="w");
                for b = 1:numBlocksForImage
                    setBlock(bimPred, [blockRows(b), blockCols(b), 1], predBlocks(coreRows, coreCols, :, b));
                    setBlock(bimGT,   [blockRows(b), blockCols(b), 1], gtBlocks(coreRows, coreCols, :, b));
                end

                % gather() requires read mode after all blocks have been written
                bimPred.Mode = 'r';
                bimGT.Mode   = 'r';
                predFull = gather(bimPred);
                gtFull   = gather(bimGT);

                subplot(ceil(numShow/floor(sqrt(numShow))), floor(sqrt(numShow)), n);
                imshow(imtile(cat(4, predFull, gtFull)));
                title(sprintf('Image id %d', uniqueIds(n)));
            end
            sgtitle("Generated Images (Prediction | Ground Truth)");
        end

        function printGPUMemory(this, label)
            % Lightweight GPU memory diagnostics.
            % Call after initStorage, densification, or key training points
            % to verify memory usage is stable and not fragmenting.
            if ~this.useGPU
                fprintf('[%s] Running on CPU — no GPU memory to report.\n', label);
                return;
            end
            g = gpuDevice;
            used = g.TotalMemory - g.AvailableMemory;
            fprintf('[%s] GPU memory used: %.2f MB / %.2f MB\n', ...
                label, used/1e6, g.TotalMemory/1e6);
        end
    end

    methods (Static)
        function env = outEnv(useGPU)
            % Maps the useGPU flag to a minibatchqueue 'OutputEnvironment' value.
            if useGPU
                env = 'gpu';
            else
                env = 'cpu';
            end
        end
    end

    methods (Static, Access = private)
        function blockSize = autoSelectBlockSize(datasetPath)
            % Auto-select a block size (per dimension, searched in [64, 160])
            % that minimizes total blockedImageDatastore partial-block
            % zero-padding summed over both resolution levels.
            %
            % Canvas sizes mirror ColmapData: Full = ceil(raw/kScaleDownFactor),
            % Half = ceil(raw/(kScaleDownFactor*2)), with kScaleDownFactor = 2.
            imds   = imageDatastore(fullfile(datasetPath, 'images'));
            info   = imfinfo(imds.Files{1});
            fullHW = ceil([info.Height, info.Width] / 2);
            halfHW = ceil([info.Height, info.Width] / 4);

            blockSize = zeros(1, 2);
            for d = 1:2
                bestPad = inf;
                for b = 64:160
                    pad = (ceil(fullHW(d)/b)*b - fullHW(d)) + (ceil(halfHW(d)/b)*b - halfHW(d));
                    if pad <= bestPad   % <= prefers larger blocks (less per-block overhead)
                        bestPad = pad;
                        blockSize(d) = b;
                    end
                end
            end
            fprintf('Auto-selected block size: [%d %d] (full canvas %dx%d, half %dx%d)\n', ...
                blockSize(1), blockSize(2), fullHW(1), fullHW(2), halfHW(1), halfHW(2));
        end

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