%% Train 3D Gaussian Splatting (3DGS)
% This example shows how-to train and generate 2D images using a
% minimal implementation of 3D Gaussian splatting.
%
% GPU Target: NVIDIA RTX 4050 Laptop GPU
%   - 6 GB GDDR6 VRAM
%   - 192 GB/s memory bandwidth
%   - 2560 CUDA cores (Ada Lovelace)
%
% Image Loading Design:
%   Images are loaded as a blockedImageDatastore (ColmapData.images); each
%   read() returns one fixed-size block (blockSize + 2*overlap), not a full
%   image. Camera info is repeated once per block with cx/cy shifted to the
%   block's local origin, so combine(obj.data.images, obj.data.cameras)
%   still gives strict 1-to-1 correspondence. blockSize defaults to
%   GaussianSplatter.autoSelectBlockSize(), which picks the size (per
%   dimension, in [64, 160]) that minimizes total partial-block zero-padding
%   summed over both resolution levels.
%
% GPU-Vectorized Rendering:
%   Rasterization no longer loops per-Gaussian or per-batch-element:
%     - projectGaussiansWithCulling processes all Gaussians x all blocks in
%       the batch as [N x B] arrays in one pass (camera transform,
%       projection, Sigma_2D via an elementwise J*Sigma*J^T, depth sort),
%       scattering results into persistent buffers by pure indexing.
%     - rasterizeToGPU composites depth-sorted Gaussians in vectorized
%       chunks per tile (fused [H' x W' x K] alpha maps, cumprod
%       transmittance, one image write per tile), with a per-chunk
%       (not per-Gaussian) early-termination check once a tile saturates.
%   Because the L1/SSIM loss is computed directly on this.image/this.image_gt
%   (now block-sized), the loss is automatically scoped to the block instead
%   of the full image — increasing miniBatchSize below now simply increases
%   the number of blocks processed per training step.
%
% Multi-Resolution Schedule:
%   Training starts at half resolution (fast early convergence).
%   At levelSwitchEpoch: setLevel(1) swaps both the image and camera
%   datastores (block layout differs per resolution), then
%   updateResolution() reallocates GPU buffers to the new block canvas size.
%   Without updateResolution(), GPU buffers stay at half-res dimensions.

%% Load Data
% Specify the folder containing the SFM generated sparse 3D point cloud as
% input, generated in COLMAP format. Download example dataset from:
% https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
datasetPath  = 'C:\Source\tandt_db\tandt\train'; % Update this path
numGaussians = 2000; % More Gaussians creates sharper images, but needs more
                     % memory and has longer training time.
numImages    = 20;

%% Blocked Image / Tile Settings
% Block/tile size used both by ColmapData's blockedImageDatastore (dataset
% block grid) and by GaussianSplatter.createImage's internal tile-culling
% loop. overlapSize adds a context halo (BorderSize) around each block;
% it does NOT change the block grid, so it has no effect on padding —
% only blockSize does.
%
% Leaving blockSize empty lets GaussianSplatter.autoSelectBlockSize() pick
% the size (per dimension, in [64, 160]) that minimizes total partial-block
% zero-padding summed over both resolution levels.
blockSize   = [];
overlapSize = [8, 8];

%% Define Learnable Parameters
% Construct object to load data and create learnable parameters.
obj = GaussianSplatter(datasetPath, numGaussians, numImages, blockSize, overlapSize);

%% Specify Training Options
% miniBatchSize now counts BLOCKS (not full images) per training step —
% raise it to increase the number of blocked images processed per batch.
totalNumBlocks = obj.data.images.TotalNumBlocks;
miniBatchSize = 2*totalNumBlocks/numImages;
numEpochs     = ceil(numGaussians / 20);

% Adam optimization options
learnRate     = 0.02;
learnInterval = ceil(numEpochs / 5);
gradDecay     = 1 - miniBatchSize / totalNumBlocks;
sqGradDecay   = 0.999;

%% Multi-Resolution Schedule
% Switch from half-resolution to full resolution at 30% of total epochs.
% After setLevel(1), updateResolution() must be called to reallocate GPU
% buffers (image, X, Y, T) to the new canvas dimensions. Without this,
% buffers remain at half-resolution, clipping Gaussian projections.
levelSwitchEpoch = max(1, floor(numEpochs * 0.3));

%% Train Model
% To save time, load a pretrained network by setting doTraining to false.
% To train the network yourself, set doTraining to true.
doTraining = true;

%% Create minibatchqueue
% combine() merges the blocked-image datastore with the per-block camera
% arrayDatastores. 1-to-1 correspondence is guaranteed because ColmapData
% repeats/offsets one camera entry per block, matching blockedImageDatastore's
% read order.
%
% OutputEnvironment='gpu' transfers each block directly to GPU VRAM on
% the RTX 4050, bypassing a redundant CPU staging copy.
%
% MiniBatchFormat:
%   Image:            'SSCB' (H × W × Channel × Batch)
%   Camera scalars:   'CB'   (1 × Batch — arrayDatastore adds leading dim)
%   Rcw rotation:     'SSCB' (3 × 3 × 1 × Batch)
%   tcw, twc vectors: 'SCB'  (3 × 1 × Batch)
ds = combine(obj.data.images, obj.data.cameras);
mbq = minibatchqueue(ds, ...
    'MiniBatchSize',    miniBatchSize, ...
    'PartialMiniBatch', 'discard', ...
    'OutputEnvironment',GaussianSplatter.outEnv(obj.useGPU), ...
    'MiniBatchFormat',  ["SSCB","CB","CB","CB","CB","CB","CB","CB","CB","CB","CB","SSCB","SCB","SCB"]);

%% Adaptive Densification Settings
enableAdaptiveDensification = true;
densifyInterval = ceil(numEpochs / 25);
prunningRatio   = 0.05;

%% Initialize Adam Optimizer State
avgGrad   = [];
avgSqGrad = [];

%% Training Progress Bookkeeping
numIterationsPerEpoch = ceil(totalNumBlocks / miniBatchSize);
numIterations         = numEpochs * numIterationsPerEpoch;

if doTraining
    monitor = trainingProgressMonitor( ...
        'Metrics', "Loss", ...
        'Info',    "Epoch", ...
        'XLabel',  "Iteration");
end

%% Initialize Preview Render Index
% Lock onto a single, unique blockId (not the reused source-image id) so
% the live preview always shows the same physical block across epochs.
blockIdxToShow = 1;

%% Custom Training Loop
if doTraining
    iteration = 0;
    epoch     = 0;

    while epoch < numEpochs && ~monitor.Stop
        epoch = epoch + 1;

        % -----------------------------------------------------------------
        % Multi-resolution schedule.
        % Step 1: setLevel(1) swaps obj.data.images/cameras from the Half
        %         pair to the Full pair (block layout differs per level,
        %         so images and cameras must switch together).
        % Step 2: updateResolution() reads the new block canvas size from
        %         the datastore and reallocates GPU buffers (image, X, Y, T).
        %         Without this call, GPU buffers stay at half-res size.
        % Step 3: Rebuild ds and mbq so they reference the new datastores,
        %         and recompute totalNumBlocks/gradDecay (block count per
        %         image differs between resolution levels).
        % -----------------------------------------------------------------
        if epoch == levelSwitchEpoch
            obj.data.setLevel(1);
            obj.updateResolution();
            ds  = combine(obj.data.images, obj.data.cameras);
            mbq = minibatchqueue(ds, ...
                'MiniBatchSize',    miniBatchSize, ...
                'PartialMiniBatch', 'discard', ...
                'OutputEnvironment',GaussianSplatter.outEnv(obj.useGPU), ...
                'MiniBatchFormat',  ["SSCB","CB","CB","CB","CB","CB","CB","CB","CB","CB","CB","SSCB","SCB","SCB"]);
            totalNumBlocks        = obj.data.images.TotalNumBlocks;
            gradDecay             = 1 - miniBatchSize / totalNumBlocks;
            numIterationsPerEpoch = ceil(totalNumBlocks / miniBatchSize);
            fprintf('Switched to full-resolution training at epoch %d.\n', epoch);

            % Print GPU Memory
            obj.printGPUMemory(sprintf('[Resolution Switch] Epoch %d', epoch));
        end

        % Shuffle data at the start of each epoch
        shuffle(mbq);

        while hasdata(mbq) && ~monitor.Stop
            iteration = iteration + 1;

            % Fetch next mini-batch. Data arrives on GPU (OutputEnvironment='gpu').
            % Camera scalars arrive as 'CB' (1×B); squeeze removes leading dim.
            % blockRow/blockCol are only needed for post-training stitching,
            % so they're ignored (~) in the hot training loop.
            [obj.image_gt, camId, camBlockId, ~, ~, camW, camH, camFx, camFy, camCx, camCy, ...
             obj.camera.Rcw, obj.camera.tcw, obj.camera.twc] = next(mbq);

            obj.camera.id      = squeeze(camId);
            obj.camera.blockId = squeeze(camBlockId);
            obj.camera.width   = squeeze(camW);
            obj.camera.height  = squeeze(camH);
            obj.camera.fx      = squeeze(camFx);
            obj.camera.fy      = squeeze(camFy);
            obj.camera.cx      = squeeze(camCx);
            obj.camera.cy      = squeeze(camCy);

            if iteration == 1
                % Initialize GPU storage once on the first iteration
                obj.initStorage(miniBatchSize);

                % Print GPU Memory
                obj.printGPUMemory('[Iteration 1] After GPU allocation');
            end

            % Forward pass, loss computation, and gradient calculation
            [loss, grads] = dlfeval(@obj.modelStep, obj.params);

            % Update learnable parameters with Adam optimizer
            [obj.params, avgGrad, avgSqGrad] = adamupdate( ...
                obj.params, grads, avgGrad, avgSqGrad, ...
                iteration, learnRate, gradDecay, sqGradDecay);

            % Extract scalar loss for logging
            loss = extractdata(loss);

            % Update training progress monitor
            recordMetrics(monitor, iteration, Loss=loss);
            updateInfo(monitor, Epoch=epoch + " of " + numEpochs);
            monitor.Progress = min(max(100 * iteration / numIterations,0),100);

            % Visualize the current render for a fixed reference block
            idx = find(extractdata(obj.camera.blockId) == blockIdxToShow);
            if ~isempty(idx)
                figure(1);
                imshow(extractdata(obj.image(:,:,:,idx)));
                title(sprintf('Epoch %d | Loss: %.4f', epoch, loss));
                drawnow;

                % Print GPU Memory
                obj.printGPUMemory(sprintf('[Periodic Monitoring] Epoch %d', epoch));
            end
        end

        % -----------------------------------------------------------------
        % Adaptive densification: prune dead Gaussians and clone/split
        % high-gradient ones to increase scene detail over time.
        % -----------------------------------------------------------------
        if enableAdaptiveDensification && ...
                mod(epoch, densifyInterval) == 0 && ...
                epoch > 1 && epoch < numEpochs
            obj.printGPUMemory(sprintf('[Before Densify] Epoch %d', epoch));
            obj.pruneAndDensify(avgGrad, avgSqGrad, prunningRatio);
            obj.printGPUMemory(sprintf('[After Densify] Epoch %d', epoch));
        end

        % Decay learning rate at regular intervals
        if mod(epoch, learnInterval) == 0 && epoch > 1 && epoch < numEpochs
            learnRate = 0.5 * learnRate;
        end
    end

    % Save trained Gaussian parameters to disk
    obj.saveGaussians("gaussians.mat");
end

%% Preview Results
% Render numGenImages full images, each stitched back together from its
% blocks, comparing prediction vs ground truth — rather than showing
% individual (scrambled, out-of-order) blocks.
figure(2);
numGenImages = 8;
load("gaussians.mat");
obj.previewResults(params, numGenImages);