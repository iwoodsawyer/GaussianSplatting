classdef GaussianSplatter < handle
    % GaussianSplatter
    % MATLAB implementation of 3D Gaussian Splatting training loop.
    %
    % GPU Target: NVIDIA RTX 4050 Laptop GPU
    %   - 6 GB GDDR6 VRAM  |  192 GB/s bandwidth  |  2560 CUDA cores
    %
    % Blocked Image Integration:
    %   - Accepts a blockSize property that matches the tile dimensions used
    %     by the blockedImageDatastore in ColmapData. The tile-aware rasterizer
    %     in createImage partitions the image into blockSize tiles and culls
    %     each Gaussian to only the tiles its bounding box overlaps, reducing
    %     redundant GPU work compared to a flat per-Gaussian loop.
    %
    %   - initStorage unconditionally moves all learnable parameters and
    %     working arrays to GPU on first call. Since minibatchqueue delivers
    %     data with OutputEnvironment='gpu', GPU residency is guaranteed.
    %
    %   - Early termination in createImage operates per-tile (transmittance
    %     threshold 0.01) rather than globally, giving finer-grained stopping.

    properties
        % --- Sizes ---
        numImages
        numGaussians
        miniBatchSize
        imageWidth
        imageHeight
        imageChannel

        % --- Blocked Image Tile Settings ---
        % Tile dimensions must match the BlockSize used in ColmapData's
        % blockedImageDatastore so the rasterizer partitions correctly.
        % Default: [256 256] — optimal for RTX 4050 (6 GB, 2560 CUDA cores).
        blockSize = [256, 256];

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
        scaleThresh       = single(0.1);   % Relative to scene size
        forcePruneThreshold = single(-4.0); % sigmoid(-4) ≈ 0.018 — prune near-invisible
        forceKeepThreshold  = single(2.0);  % sigmoid(2)  ≈ 0.88  — always keep opaque
        splitScaleFactor    = single(0.2);  % Scale reduction on split: log(1.6) ≈ 0.47;
                                            % 0.2 gives a gentler split for small VRAM budgets

        % --- Working Arrays (allocated in initStorage, resident on GPU) ---
        image     % Rendered output:  [H x W x 3 x B] dlarray 'SSCB'
        image_gt  % Ground truth tile: [H x W x 3 x B] dlarray 'SSCB'
        camera    % Struct of current mini-batch camera parameters
        window    % Gaussian SSIM kernel: [wS x wS x C x C] dlarray 'SSCU'
        X         % Pixel column coordinates [H x W x 1] dlarray 'SSC'
        Y         % Pixel row coordinates    [H x W x 1] dlarray 'SSC'
        XY
        T         % Per-pixel transmittance  [H x W x 1] dlarray 'SSC'
    end

    methods
        function this = GaussianSplatter(datasetPath, numGaussians, numImages, blockSize)
            % Constructor: load COLMAP data, initialise learnable parameters,
            % and create the SSIM Gaussian kernel window.
            %
            % Args:
            %   datasetPath  (string) : Path to the COLMAP dataset root.
            %   numGaussians (int)    : Number of Gaussians to initialise.
            %   numImages    (int)    : Number of training images to load.
            %   blockSize    (1x2 int): Tile [H W] matching blockedImageDatastore.
            %                          Defaults to [256 256].

            if nargin < 4 || isempty(blockSize)
                blockSize = [256, 256];
            end

            this.camera    = struct;
            this.blockSize = blockSize;

            % Store sizes
            this.datasetPath  = datasetPath;
            this.numGaussians = numGaussians;
            this.numImages    = numImages;

            % Load data and create learnable parameters.
            % miniBatchSize=2 is passed so ColmapData sizes ReadSize and
            % apply BatchSize correctly for the RTX 4050 6 GB VRAM budget.
            this.data = ColmapData(datasetPath, numGaussians, numImages, blockSize, 2);

            % Determine image (tile) dimensions from the blocked datastore
            previewTile = read(this.data.images);
            if iscell(previewTile)
                previewTile = previewTile{1};
            end
            [this.imageHeight, this.imageWidth, this.imageChannel] = size(previewTile);

            % Create learnable parameter struct (dlarray, no GPU yet)
            this.params = this.createLearnableParams(this.data.gaussians);

            % Create the 2D Gaussian kernel for the differentiable SSIM loss
            this.window = this.createWindow(this.windowSize, this.imageChannel);

            % Scene scale drives the densification scale threshold
            temp           = preview(this.data.scene_scale);
            this.sceneScale = temp{1};
        end

        function initStorage(this, miniBatchSize)
            % Allocate all working GPU arrays once on the first training
            % iteration, when image dimensions are known.
            %
            % All arrays are created directly on the GPU (gpuArray) so there
            % is no deferred CPU-to-GPU copy. On the RTX 4050, placing arrays
            % in GPU memory at construction avoids PCIe transfer latency on
            % every subsequent iteration.

            this.miniBatchSize = miniBatchSize;

            % Rendered image buffer — lives on GPU for the full training run
            this.image = dlarray( ...
                gpuArray(zeros(this.imageHeight, this.imageWidth, 3, ...
                               this.miniBatchSize, 'single')), 'SSCB');

            % Pixel coordinate grids — constant, allocated once on GPU
            [Xi, Yi] = meshgrid(1:uint32(this.imageWidth), 1:uint32(this.imageHeight));
            this.X = dlarray(gpuArray(repmat(single(Xi), 1, 1, 1)), 'SSC');
            this.Y = dlarray(gpuArray(repmat(single(Yi), 1, 1, 1)), 'SSC');

            % Per-pixel transmittance accumulator — reset to 1 per Gaussian pass
            this.T = dlarray(gpuArray(ones(this.imageHeight, this.imageWidth, 1, 'single')), 'SSC');

            % Spherical harmonic coefficient vector on GPU
            this.shToColor = dlarray(gpuArray(this.shToColor), 'SSC');

            % SSIM loss constants on GPU
            this.C1 = gpuArray(this.C1);
            this.C2 = gpuArray(this.C2);

            % Combined loss weight on GPU
            this.lambda = gpuArray(this.lambda);

            % SSIM Gaussian kernel on GPU
            this.window = gpuArray(this.window);

            % --- Move all learnable parameters to GPU ---
            % Performed unconditionally since minibatchqueue always delivers
            % data with OutputEnvironment='gpu' in the blocked image pipeline.
            fprintf('Moving learnable parameters to GPU (RTX 4050)...\n');
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

            % Combined weighted loss: λ balances pixel fidelity vs. structure
            loss = (1.0 - this.lambda) * l1_loss + this.lambda * ssim_loss;

            % Compute gradients with respect to all learnable parameters
            grads = dlgradient(loss, params);
        end

        function ssim_val = ssimLoss(this)
            % Compute the differentiable Structural Similarity (SSIM) metric.
            %
            % Uses depthwise dlconv with a separable Gaussian kernel (window)
            % to compute local means and variances. The diagonal channel
            % structure of window ensures channel independence.

            mu1 = dlconv(this.image,    this.window, 0, 'Padding', 'same');
            mu2 = dlconv(this.image_gt, this.window, 0, 'Padding', 'same');

            mu1_sq  = mu1.^2;
            mu2_sq  = mu2.^2;
            mu1_mu2 = mu1 .* mu2;

            sigma1_sq = dlconv(this.image.^2,              this.window, 0, 'Padding', 'same') - mu1_sq;
            sigma2_sq = dlconv(this.image_gt.^2,           this.window, 0, 'Padding', 'same') - mu2_sq;
            sigma12   = dlconv(this.image .* this.image_gt, this.window, 0, 'Padding', 'same') - mu1_mu2;

            % SSIM map: numerator captures cross-correlation; denominator
            % normalises by the sum of individual variances.
            ssim_map = ((single(2.0) .* mu1_mu2 + this.C1) .* (single(2.0) .* sigma12   + this.C2)) ./ ...
                       ((mu1_sq + mu2_sq + this.C1)         .* (sigma1_sq   + sigma2_sq  + this.C2));

            % Scalar SSIM: average over all spatial locations and channels
            ssim_val = mean(ssim_map, 'all');
        end

        function createImage(this, params)
            % Tile-aware Gaussian Splatting rasterizer.
            %
            % The image is partitioned into blockSize tiles. Each Gaussian is
            % only rasterized into tiles whose bounding box it overlaps,
            % reducing redundant GPU operations compared to a flat per-Gaussian
            % loop over the whole image. This is the key GPU performance
            % improvement enabled by the blocked image pipeline.
            %
            % Tile dimensions (blockSize) must match the BlockSize used in the
            % blockedImageDatastore so the rasterizer and datastore are aligned.
            %
            % Early termination is applied per-tile: once a tile's accumulated
            % transmittance drops below 0.01 (fully opaque), no further
            % Gaussians are composited into that tile.

            % Strip dlarray dimension labels for indexing
            this.camera = structfun(@stripdims, this.camera, 'UniformOutput', false);

            % Reset rendered image to zero for this mini-batch
            this.image(:) = 0;

            % Project all Gaussians with frustum culling and depth-sort
            [u, v, alphas, Cov, colors] = this.projectGaussiansWithCulling(params);

            % Tile dimensions from the blocked image configuration
            blockH = this.blockSize(1);
            blockW = this.blockSize(2);

            numTilesH = ceil(this.imageHeight / blockH);
            numTilesW = ceil(this.imageWidth  / blockW);

            for b = 1:this.miniBatchSize
                % Reset transmittance to 1 (fully transparent) for each image
                this.T(:) = 1;

                % Compute singular value decomposition of 2D projected covariances
                [U, S, ~] = this.svd3(shiftdim(stripdims(Cov(1:length(alphas),:,:,b)), 1));

                % 3-sigma pixel radii: half-extents of each Gaussian's screen footprint
                radiusPixels = single(3.0) .* S;
                radiusPixels = U(1:2,:,:) .* radiusPixels;
                radiusPixels = ceil(extractdata(squeeze(vecnorm(radiusPixels, 2, 2))))';

                % Screen-space bounding boxes for all Gaussians
                umin = max(1,               floor(extractdata(u(1:length(alphas),:,:,b))) - radiusPixels(:,1,:));
                umax = min(this.imageWidth,  ceil(extractdata(u(1:length(alphas),:,:,b))) + radiusPixels(:,1,:));
                vmin = max(1,               floor(extractdata(v(1:length(alphas),:,:,b))) - radiusPixels(:,2,:));
                vmax = min(this.imageHeight, ceil(extractdata(v(1:length(alphas),:,:,b))) + radiusPixels(:,2,:));

                % Discard degenerate bounding boxes (zero or negative area)
                valid_indices = find((umin <= umax) & (vmin <= vmax));

                % ----------------------------------------------------------
                % Tile-aware rasterization loop
                % For each tile, cull to the subset of Gaussians that
                % overlap this tile. This is the core GPU efficiency gain:
                % a Gaussian only contributes compute to tiles it covers.
                % ----------------------------------------------------------
                for th = 1:numTilesH
                    tileVmin = (th - 1) * blockH + 1;
                    tileVmax = min(th * blockH, this.imageHeight);

                    for tw = 1:numTilesW
                        tileUmin = (tw - 1) * blockW + 1;
                        tileUmax = min(tw * blockW, this.imageWidth);

                        % Cull: keep only Gaussians whose bounding box
                        % intersects this tile's pixel rectangle.
                        tileIdx = valid_indices( ...
                            umin(valid_indices) <= tileUmax & ...
                            umax(valid_indices) >= tileUmin & ...
                            vmin(valid_indices) <= tileVmax & ...
                            vmax(valid_indices) >= tileVmin);

                        for i = 1:length(tileIdx)
                            k = tileIdx(i);

                            % Clamp Gaussian bounding box to tile boundaries
                            umin_k = max(umin(k), tileUmin);
                            umax_k = min(umax(k), tileUmax);
                            vmin_k = max(vmin(k), tileVmin);
                            vmax_k = min(vmax(k), tileVmax);

                            % Gaussian falloff: weight each pixel by its
                            % Mahalanobis distance from the Gaussian centre
                            alphaT = stripdims([ ...
                                reshape(this.X(vmin_k:vmax_k, umin_k:umax_k, 1), [], 1) - u(k,:,:,b), ...
                                reshape(this.Y(vmin_k:vmax_k, umin_k:umax_k, 1), [], 1) - v(k,:,:,b)]) ...
                                * (U(1:2,:,k) ./ max(S(:,:,k), 1e-6));
                            alphaT = dlarray(reshape(sum(alphaT .* alphaT, 2), ...
                                vmax_k - vmin_k + 1, umax_k - umin_k + 1), 'SSC');
                            alphaT = exp(-single(0.5) .* alphaT);
                            alphaT = alphas(k,:,:,b) .* alphaT;

                            % Painter's algorithm: modulate by accumulated transmittance
                            alphaT = this.T(vmin_k:vmax_k, umin_k:umax_k, 1, 1) .* alphaT;

                            % Alpha-composite into the rendered image (per channel)
                            this.image(vmin_k:vmax_k, umin_k:umax_k, 1, b) = ...
                                this.image(vmin_k:vmax_k, umin_k:umax_k, 1, b) + alphaT .* colors(k,:,1,b);
                            this.image(vmin_k:vmax_k, umin_k:umax_k, 2, b) = ...
                                this.image(vmin_k:vmax_k, umin_k:umax_k, 2, b) + alphaT .* colors(k,:,2,b);
                            this.image(vmin_k:vmax_k, umin_k:umax_k, 3, b) = ...
                                this.image(vmin_k:vmax_k, umin_k:umax_k, 3, b) + alphaT .* colors(k,:,3,b);

                            % Update transmittance: subtract composited alpha
                            this.T(vmin_k:vmax_k, umin_k:umax_k, 1, 1) = ...
                                this.T(vmin_k:vmax_k, umin_k:umax_k, 1, 1) - alphaT;

                            % Per-tile early termination: stop once this tile
                            % is fully opaque (transmittance < 0.01)
                            if max(this.T(tileVmin:tileVmax, tileUmin:tileUmax), [], 'all') < 0.01
                                break;
                            end
                        end
                    end
                end
            end
        end

        function [u, v, alphas, Cov, colors] = projectGaussiansWithCulling(this, params)
            % Project 3D Gaussians to screen space with frustum culling.
            %
            % Projects Gaussian centres through the camera model, applies
            % frustum culling (depth and screen margin), sorts surviving
            % Gaussians by depth (Painter's Algorithm — back-to-front),
            % and evaluates covariance, opacity, and spherical harmonic color.

            % Transform world-space positions to camera space
            gaussians = pagemtimes( ...
                repmat(params.pws, 1, 1, this.miniBatchSize), 'none', ...
                this.camera.Rcw, 'transpose');
            gaussians = gaussians + repmat( ...
                pagetranspose(reshape(this.camera.tcw, 3, 1, this.miniBatchSize)), ...
                size(params.pws,1), 1, 1);

            % Perspective projection: divide X, Y by Z, then apply intrinsics
            inv_z = single(1.0) ./ max(gaussians(:,3,:), single(1e-6));
            gaussians(:,1,:) = gaussians(:,1,:) .* inv_z .* reshape(this.camera.fx, 1, 1, this.miniBatchSize) ...
                             + reshape(this.camera.cx, 1, 1, this.miniBatchSize);
            gaussians(:,2,:) = gaussians(:,2,:) .* inv_z .* reshape(this.camera.fy, 1, 1, this.miniBatchSize) ...
                             + reshape(this.camera.cy, 1, 1, this.miniBatchSize);

            % Frustum culling: keep Gaussians within depth bounds and a
            % 50% screen-margin safety window to handle large splats.
            valid = (gaussians(:,3,:) > single(0.2))                              & ...
                    (gaussians(:,3,:) < single(100.0))                             & ...
                    (gaussians(:,1,:) > -single(this.imageWidth)  * single(0.5))  & ...
                    (gaussians(:,1,:) <  single(this.imageWidth)  * single(1.5))  & ...
                    (gaussians(:,2,:) > -single(this.imageHeight) * single(0.5))  & ...
                    (gaussians(:,2,:) <  single(this.imageHeight) * single(1.5));

            % Return empty outputs if no Gaussians survive culling
            nr_valid = sum(valid, 1);
            if ~any(nr_valid)
                u      = [];
                v      = [];
                alphas = [];
                Cov    = [];
                colors = [];
                return;
            end

            % Preallocate outputs sized to the maximum surviving Gaussian count
            numValid = extractdata(max(nr_valid));
            u      = dlarray(zeros(numValid, 1, 1, this.miniBatchSize, like=this.shToColor), 'SSCB');
            v      = dlarray(zeros(numValid, 1, 1, this.miniBatchSize, like=this.shToColor), 'SSCB');
            alphas = dlarray(zeros(numValid, 1, 1, this.miniBatchSize, like=this.shToColor), 'SSCB');
            Cov    = dlarray(zeros(numValid, 3, 3, this.miniBatchSize, like=this.shToColor), 'SSSB');
            colors = dlarray(zeros(numValid, 1, 3, this.miniBatchSize, like=this.shToColor), 'SSCB');
            Sh     = dlarray(zeros(numValid, length(this.shToColor),   like=this.shToColor), 'SSC');

            for b = 1:this.miniBatchSize
                % Sort surviving Gaussians back-to-front by depth (Painter's Algorithm)
                [~, sortIdx] = sort(gaussians(valid(:,:,b), 3, b), 'descend');
                validIdx     = find(extractdata(valid(:,:,b)));
                sortIdx      = validIdx(sortIdx);
                numIdx       = length(sortIdx);

                % Screen-space positions
                u(1:numIdx,:,:,b) = gaussians(sortIdx, 1, b);
                v(1:numIdx,:,:,b) = gaussians(sortIdx, 2, b);

                % Opacity: sigmoid activation of raw logit
                alphas(1:numIdx,:,:,b) = single(1.0) ./ (single(1.0) + exp(-params.alphas_raw(sortIdx)));

                % Normalize quaternions to unit length before rotation conversion
                quat = params.rots_raw(sortIdx,:);
                quat = quat ./ max(vecnorm(quat, 2, 2), 1e-6);

                % Convert quaternions to rotation matrices (stored in Cov temporarily)
                Cov(1:numIdx,1,1,b) = single(1.0) - single(2.0)*(quat(:,3).*quat(:,3) + quat(:,4).*quat(:,4));
                Cov(1:numIdx,1,2,b) = single(2.0)*(quat(:,2).*quat(:,3) - quat(:,4).*quat(:,1));
                Cov(1:numIdx,1,3,b) = single(2.0)*(quat(:,2).*quat(:,4) + quat(:,3).*quat(:,1));
                Cov(1:numIdx,2,1,b) = single(2.0)*(quat(:,2).*quat(:,3) + quat(:,4).*quat(:,1));
                Cov(1:numIdx,2,2,b) = single(1.0) - single(2.0)*(quat(:,2).*quat(:,2) + quat(:,4).*quat(:,4));
                Cov(1:numIdx,2,3,b) = single(2.0)*(quat(:,3).*quat(:,4) - quat(:,2).*quat(:,1));
                Cov(1:numIdx,3,1,b) = single(2.0)*(quat(:,2).*quat(:,4) - quat(:,3).*quat(:,1));
                Cov(1:numIdx,3,2,b) = single(2.0)*(quat(:,3).*quat(:,4) + quat(:,2).*quat(:,1));
                Cov(1:numIdx,3,3,b) = single(1.0) - single(2.0)*(quat(:,2).*quat(:,2) + quat(:,3).*quat(:,3));

                % Combine camera extrinsic rotation with Gaussian rotation
                Cov(:,:,:,b) = shiftdim(pagemtimes( ...
                    this.camera.Rcw(:,:,b), shiftdim(stripdims(Cov(:,:,:,b)), 1)), 2);

                % Apply diagonal scale matrix and project to screen pixels
                % Scale is exponentiated from log-space raw parameter.
                % Clamp to [-10, 10] to prevent numerical overflow.
                Cov(1:numIdx,:,1,b) = Cov(1:numIdx,:,1,b) .* exp(min(max(params.scales_raw(sortIdx,1), single(-10)), single(10))) .* inv_z(sortIdx,:,b) .* this.camera.fx(b);
                Cov(1:numIdx,:,2,b) = Cov(1:numIdx,:,2,b) .* exp(min(max(params.scales_raw(sortIdx,2), single(-10)), single(10))) .* inv_z(sortIdx,:,b) .* this.camera.fx(b);
                Cov(1:numIdx,:,3,b) = Cov(1:numIdx,:,3,b) .* exp(min(max(params.scales_raw(sortIdx,3), single(-10)), single(10))) .* inv_z(sortIdx,:,b) .* this.camera.fx(b);

                % Unit-normalise 3D position for spherical harmonic evaluation
                pos = gaussians(sortIdx,:,b);
                pos = pos ./ max(vecnorm(pos, 2, 2), 1e-6);

                % Evaluate second-order spherical harmonics for RGB color
                Sh(1:numIdx,1) = this.shToColor(1);
                Sh(1:numIdx,2) = this.shToColor(2) .* (-pos(:,1));
                Sh(1:numIdx,3) = this.shToColor(3) .* (-pos(:,2));
                Sh(1:numIdx,4) = this.shToColor(4) .* ( pos(:,3));
                Sh(1:numIdx,5) = this.shToColor(5) .* ( pos(:,1) .* pos(:,2));
                Sh(1:numIdx,6) = this.shToColor(6) .* (-pos(:,1) .* pos(:,3));
                Sh(1:numIdx,7) = this.shToColor(7) .* (-pos(:,2) .* pos(:,3));
                Sh(1:numIdx,8) = this.shToColor(8) .* (single(3.0) .* pos(:,3).*pos(:,3) - single(1.0));
                Sh(1:numIdx,9) = this.shToColor(9) .* (pos(:,1).*pos(:,1) - pos(:,2).*pos(:,2));

                % Clamp colors to valid [0, 1] range after SH expansion
                colors(1:length(sortIdx),:,1,b) = max(min(single(0.5) + sum(Sh(1:numIdx,:) .* params.shs(sortIdx,:,1), 2), single(1.0)), single(0.0));
                colors(1:length(sortIdx),:,2,b) = max(min(single(0.5) + sum(Sh(1:numIdx,:) .* params.shs(sortIdx,:,2), 2), single(1.0)), single(0.0));
                colors(1:length(sortIdx),:,3,b) = max(min(single(0.5) + sum(Sh(1:numIdx,:) .* params.shs(sortIdx,:,3), 2), single(1.0)), single(0.0));
            end
        end

        function pruneAndDensify(this, avgGrad, prunningRatio)
            % Adaptive densification: prune low-contribution Gaussians and
            % replace them with clones or splits of high-gradient Gaussians.
            %
            % Pruning strategy:
            %   - Low opacity × low gradient → dead weight → prune.
            %   - Force-prune if opacity logit < forcePruneThreshold (≈0.018).
            %   - Force-keep if opacity logit > forceKeepThreshold  (≈0.88).
            %
            % Densification strategy:
            %   - Clone: duplicate a high-gradient Gaussian at the same location.
            %   - Split: reduce scale by splitScaleFactor on both halves to
            %     represent a complex region with two smaller Gaussians.

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

                % Remove any index that is already scheduled for pruning
                cloneIdx = setdiff(extractdata(cloneIdx), extractdata(pruneIdx), 'stable');
                cloneIdx = cloneIdx(1:numPrune);

                % Decide split vs. clone based on current scale
                shouldSplit = max(exp(this.params.scales_raw), [], 2) > ...
                              this.scaleThresh * this.sceneScale;

                % Replace pruned slots with clones (or splits) of selected Gaussians
                this.params.pws(pruneIdx,:)        = this.params.pws(cloneIdx,:);
                this.params.alphas_raw(pruneIdx,:) = this.params.alphas_raw(cloneIdx,:);
                this.params.scales_raw(pruneIdx,:) = this.params.scales_raw(cloneIdx,:) ...
                                                     - shouldSplit(cloneIdx,:) .* this.splitScaleFactor;
                % Apply matching scale reduction to the original clone slot
                this.params.scales_raw(cloneIdx,:) = this.params.scales_raw(cloneIdx,:) ...
                                                     - shouldSplit(cloneIdx,:) .* this.splitScaleFactor;
                this.params.rots_raw(pruneIdx,:)   = this.params.rots_raw(cloneIdx,:);
                this.params.shs(pruneIdx,:)        = this.params.shs(cloneIdx,:);

                fprintf('Densification: %d clones, %d splits.\n', ...
                    sum(~shouldSplit(cloneIdx,:)), sum(shouldSplit(cloneIdx,:)));
            end
        end

        function saveGaussians(this, filename)
            % Gather GPU arrays back to CPU and save learnable parameters.
            params = gather(this.params); %#ok<PROPLC>
            save(filename, "params");
        end
    end

    methods (Static, Access = private)
        function paramStruct = createLearnableParams(gaussians)
            % Convert Gaussian struct of doubles to a struct of dlarrays.
            % All parameters start on CPU; initStorage migrates them to GPU.

            paramStruct.pws = dlarray(single(gaussians.pws));

            % Reshape SH coefficients to [N x 9 x 3]: N Gaussians,
            % 9 SH basis coefficients, 3 colour channels.
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

        function window_kernel = createWindow(window_size, channel)
            % Build a depthwise 2D Gaussian kernel for the SSIM loss.
            %
            % A separable 1D Gaussian (sigma=1.5) is outer-producted to form
            % the 2D kernel. The diagonal [window_size x window_size x C x C]
            % structure ensures each channel is filtered independently.

            sigma    = single(1.5);
            coords   = 0:single(window_size - 1);
            center   = floor(window_size / 2);
            gauss    = exp(-(coords - center).^2 / (2 * sigma^2));
            gauss    = gauss / sum(gauss);  % Normalise to unit sum

            % Outer product → 2D Gaussian footprint
            gauss_2d = gauss(:) * gauss(:)';

            % Depthwise kernel: non-zero only on the diagonal (c_in == c_out)
            window_kernel = zeros(window_size, window_size, channel, channel, 'single');
            for c = 1:channel
                window_kernel(:,:,c,c) = gauss_2d;
            end

            % Wrap as dlarray with format SSCU (Spatial, Spatial, Channel_In, Filter)
            window_kernel = dlarray(window_kernel, 'SSCU');
        end

        function [U, S, V] = svd3(A)
            % SVD3 — Differentiable SVD for batches of 3×3 matrices.
            %
            %   [U, S, V] = svd3(A) returns U, S, V such that A = U * S * V'.
            %
            %   Input:
            %       A — dlarray of size 3 × 3 × N (batch of 3×3 matrices)
            %
            %   Outputs:
            %       U — Left singular vectors  (3 × 3 × N)
            %       S — Singular values as diagonal matrices (3 × 3 × N)
            %       V — Right singular vectors (3 × 3 × N)
            %
            %   Method:
            %       One-sided Jacobi iteration (5 sweeps). Sufficient for
            %       single-precision 3×3 matrices. Supports dlgradient.

            batch_size = size(A, 3);

            % Initialise V as batch of identity matrices
            V = dlarray(repmat(eye(3, like=extractdata(A(1))), 1, 1, batch_size));

            % 5 Jacobi sweeps over all off-diagonal pairs
            num_iters = 5;
            for iter = 1:num_iters
                [A, V] = jacobi_step(A, V, 1, 2);
                [A, V] = jacobi_step(A, V, 1, 3);
                [A, V] = jacobi_step(A, V, 2, 3);
            end

            % Singular values: L2 norm of each column of A after convergence
            S = sqrt(sum(A .* A, 1));

            % Left singular vectors: normalise columns of A
            U = A ./ max(S, 1e-6);

            function [A, V] = jacobi_step(A, V, p, q)
                % Apply a Jacobi rotation to annihilate element (p,q).

                Ap = A(:,p,:);
                Aq = A(:,q,:);

                % Elements of the 2×2 Gram sub-matrix
                a = sum(Ap .* Ap, 1);
                b = sum(Aq .* Aq, 1);
                d = sum(Ap .* Aq, 1);

                % Rotation angle to zero out d
                numer = single(2.0) * d;
                denom = a - b;

                % Guard against isotropic case (atan2 of [0,0] is undefined)
                is_safe   = (abs(numer) > single(1e-6)) | (abs(denom) > single(1e-6));
                safe_numer = numer + single(~is_safe);
                safe_denom = denom + single(~is_safe);

                theta = single(0.5) .* atan2(safe_numer, safe_denom);
                theta = theta .* single(is_safe);  % Zero angle for isotropic case

                c = cos(theta);
                s = sin(theta);

                % Reshape scalars for batch broadcasting
                c = reshape(c, 1, 1, []);
                s = reshape(s, 1, 1, []);

                % Apply Givens rotation to columns p and q of A
                A(:,p,:) =  c .* Ap + s .* Aq;
                A(:,q,:) = -s .* Ap + c .* Aq;

                % Accumulate rotations in V
                Vp = V(:,p,:);
                Vq = V(:,q,:);
                V(:,p,:) =  c .* Vp + s .* Vq;
                V(:,q,:) = -s .* Vp + c .* Vq;
            end
        end
    end
end