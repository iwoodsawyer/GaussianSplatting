%% Train 3D Gaussian Splatting (3DGS)
% This example shows how-to train and generate 2D images using a
% minimal implementation of 3D Gaussian splatting.
%
% GPU Target: NVIDIA RTX 4050 Laptop GPU
%   - 6 GB GDDR6 VRAM
%   - 192 GB/s memory bandwidth
%   - 2560 CUDA cores (Ada Lovelace)
%
% Blocked Image Pipeline:
%   - Ground-truth images are loaded as blockedImage objects using the
%     Image Processing Toolbox, decomposed into 256x256 tiles.
%   - A two-level multi-resolution pyramid is built with makeMultiLevel2D.
%     Training starts at half-resolution (Level 2) for fast early convergence,
%     then switches to full resolution (Level 1) at 30% of total epochs.
%   - blockedImageDatastore streams tiles directly to the GPU via
%     minibatchqueue with OutputEnvironment='gpu', eliminating full-image
%     CPU-to-GPU transfers.

%% Load Data
% Specify the folder containing the SFM generated sparse 3D point cloud as
% input, generated in COLMAP format. Download example dataset from:
% https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
datasetPath  = 'C:\Source\tandt_db\tandt\train'; % Update this path
numGaussians = 2000; % More Gaussians creates sharper images, but needs more
                     % memory and has longer training time.
numImages    = 20;

%% Blocked Image Settings
% Tile size for blockedImage decomposition.
% 256x256 is optimal for the RTX 4050 Laptop (6 GB VRAM, 2560 CUDA cores):
%   - Fits multiple tiles per GPU warp efficiently.
%   - Keeps per-iteration VRAM footprint well under the 6 GB ceiling.
%   - Aligns with CUDA memory transaction width for Ada Lovelace.
blockSize = [256, 256];

%% Define Learnable Parameters
% Construct object to load data and create learnable parameters.
% blockSize is passed so the tile-aware rasterizer uses the same tile
% dimensions as the blockedImageDatastore.
obj = GaussianSplatter(datasetPath, numGaussians, numImages, blockSize);

%% Specify Training Options
% Train for numEpochs epochs with a mini-batch size of 2.
% miniBatchSize=2 keeps VRAM usage balanced on the RTX 4050 6 GB budget:
% each 256x256x3 single-precision tile pair occupies ~1.5 MB, leaving
% headroom for learnable parameters and gradient buffers.
miniBatchSize = 2;
numEpochs     = ceil(numGaussians / 20);

% Specify the options for Adam optimization.
learnRate     = 0.02;
learnInterval = ceil(numEpochs / 5);
gradDecay     = 1 - miniBatchSize / numImages;
sqGradDecay   = 0.999;

%% Multi-Resolution Schedule
% Switch from half-resolution (Level 2) to full resolution (Level 1) at
% 30% through training. Early epochs converge faster at lower resolution;
% later epochs refine fine details at full resolution.
levelSwitchEpoch = max(1, floor(numEpochs * 0.3));

%% Train Model
% Train the 3D Gaussian splat model using a custom training loop.
%
% Training is computationally expensive and can take hours. To save time
% while running this example, load a pretrained network by setting
% doTraining to false. To train the network yourself, set doTraining to true.
doTraining = true;

%% Create minibatchqueue
% Create a minibatchqueue that processes and manages mini-batches of image
% tiles and camera data during training.
%
% The combined datastore merges the blockedImageDatastore (image tiles) with
% the camera label arrayDatastores. OutputEnvironment='gpu' transfers each
% block directly to GPU memory, bypassing a redundant CPU staging copy.
ds = combine(obj.data.images, obj.data.cameras);
mbq = minibatchqueue(ds, ...
    'MiniBatchSize',    miniBatchSize, ...
    'PartialMiniBatch', 'discard', ...
    'OutputEnvironment','gpu', ...
    'MiniBatchFormat',  ["SSCB","B","B","B","B","B","B","B","SSB","SB","SB"]);

%% Adaptive Densification Settings
enableAdaptiveDensification = true;
densifyInterval = ceil(numEpochs / 25);
prunningRatio   = 0.05;

%% Initialize Adam Optimizer State
avgGrad   = [];
avgSqGrad = [];

%% Training Progress Bookkeeping
numIterationsPerEpoch = ceil(numImages / miniBatchSize);
numIterations         = numEpochs * numIterationsPerEpoch;

if doTraining
    monitor = trainingProgressMonitor( ...
        'Metrics', "Loss", ...
        'Info',    "Epoch", ...
        'XLabel',  "Iteration");
end

%% Initialize Preview Render Index
imageIdxToShow = preview(obj.data.cameras);
imageIdxToShow = imageIdxToShow{1};

%% Custom Training Loop
% For each epoch, shuffle the image tiles and camera data and loop over
% mini-batches. Adaptive densification prunes low-contribution Gaussians
% and clones/splits high-gradient ones at regular intervals.
if doTraining
    iteration = 0;
    epoch     = 0;

    while epoch < numEpochs && ~monitor.Stop
        epoch = epoch + 1;

        % -----------------------------------------------------------------
        % Multi-resolution schedule: switch to full-resolution at the
        % configured epoch threshold.
        % -----------------------------------------------------------------
        if epoch == levelSwitchEpoch
            obj.data.images.Level = 1;  % Level 1 = finest resolution
            fprintf('Switched blockedImageDatastore to full resolution at epoch %d.\n', epoch);
        end

        % Shuffle data at the start of each epoch
        shuffle(mbq);

        while hasdata(mbq) && ~monitor.Stop
            iteration = iteration + 1;

            % Fetch next mini-batch of image tiles and camera parameters.
            % Data arrives on GPU due to OutputEnvironment='gpu'.
            [obj.image_gt, obj.camera.id, obj.camera.width, obj.camera.height, ...
             obj.camera.fx, obj.camera.fy, obj.camera.cx, obj.camera.cy, ...
             obj.camera.Rcw, obj.camera.tcw, obj.camera.twc] = next(mbq);

            if iteration == 1
                % Initialize GPU storage on first iteration, once data
                % dimensions are known from the first mini-batch.
                obj.initStorage(miniBatchSize);
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
            monitor.Progress = 100 * iteration / numIterations;

            % Visualize the current render for a fixed reference view
            idx = find(extractdata(obj.camera.id) == imageIdxToShow);
            if ~isempty(idx)
                figure(1);
                imshow(extractdata(obj.image(:,:,:,idx)));
                title(sprintf('Epoch %d | Loss: %.4f', epoch, loss));
                drawnow;
            end
        end

        % -----------------------------------------------------------------
        % Adaptive densification: prune dead Gaussians and clone/split
        % high-gradient ones to increase scene detail over time.
        % -----------------------------------------------------------------
        if enableAdaptiveDensification && ...
                mod(epoch, densifyInterval) == 0 && ...
                epoch > 1 && epoch < numEpochs
            obj.pruneAndDensify(avgGrad, prunningRatio);
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
% Render a grid of generated vs. ground-truth image pairs after training.
figure(2);
numGenImages = 8;
load("gaussians.mat");
shuffle(mbq);

for iteration = 1:ceil(numGenImages / miniBatchSize)
    [obj.image_gt, obj.camera.id, obj.camera.width, obj.camera.height, ...
     obj.camera.fx, obj.camera.fy, obj.camera.cx, obj.camera.cy, ...
     obj.camera.Rcw, obj.camera.tcw, obj.camera.twc] = next(mbq);

    if iteration == 1 && isempty(obj.image)
        obj.initStorage(miniBatchSize);
    end
    obj.createImage(params);

    genImages = cat(4, gather(extractdata(obj.image)), ...
                       gather(extractdata(obj.image_gt)));
    subplot(2, 2, iteration);
    imshow(imtile(genImages));
end
sgtitle("Generated Images");