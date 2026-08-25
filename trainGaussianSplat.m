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
%   Ground-truth images are loaded as blockedImage objects using the
%   Image Processing Toolbox, decomposed into 256x256 tiles.
%   Two single-level blockedImageDatastores are pre-built:
%     imagesHalf : half-resolution for fast early-epoch convergence
%     imagesFull : full resolution for fine-detail refinement
%   Training starts at half-resolution and switches at levelSwitchEpoch.
%   blockedImageDatastore.read() returns cell arrays; a transform()
%   unwraps each cell to a plain H×W×C single array before combine().
%   minibatchqueue with OutputEnvironment='gpu' transfers tiles
%   directly to GPU VRAM, eliminating CPU staging copies.

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
% 256x256 is optimal for the RTX 4050 Laptop (6 GB VRAM, 2560 CUDA cores).
blockSize = [256, 256];

%% Define Learnable Parameters
% Construct object to load data and create learnable parameters.
% blockSize is passed so the tile-aware rasterizer uses the same tile
% dimensions as the blockedImageDatastore.
obj = GaussianSplatter(datasetPath, numGaussians, numImages, blockSize);

%% Specify Training Options
% Train for numEpochs epochs with a mini-batch size of 2.
% miniBatchSize=2 keeps VRAM usage balanced on the RTX 4050 6 GB budget.
miniBatchSize = 2;
numEpochs     = ceil(numGaussians / 20);

% Adam optimization options
learnRate     = 0.02;
learnInterval = ceil(numEpochs / 5);
gradDecay     = 1 - miniBatchSize / numImages;
sqGradDecay   = 0.999;

%% Multi-Resolution Schedule
% Switch from half-resolution to full resolution at 30% of total epochs.
% Early epochs converge faster at lower resolution; later epochs refine
% fine details at full resolution.
levelSwitchEpoch = max(1, floor(numEpochs * 0.3));

%% Train Model
% To save time, load a pretrained network by setting doTraining to false.
% To train the network yourself, set doTraining to true.
doTraining = true;

%% Create minibatchqueue
% Combine the image tile datastore with the camera label datastores.
% blockedImageDatastore.read() returns a cell array of blocks.
% ColmapData.unwrapBlockCell (applied via transform in ColmapData) converts
% each cell to a plain H×W×C single array so combine/horzcat succeeds.
%
% OutputEnvironment='gpu' transfers each block directly to GPU VRAM,
% bypassing a redundant CPU staging copy on the RTX 4050.
ds = combine(obj.data.images, obj.data.cameras);

mbq = minibatchqueue(ds, ...
    'MiniBatchSize',    miniBatchSize, ...
    'PartialMiniBatch', 'discard', ...
    'OutputEnvironment','gpu', ...
    'MiniBatchFormat',  ["SSCB","CB","CB","CB","CB","CB","CB","CB","SSCB","SCB","SCB"]);

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
if doTraining
    iteration = 0;
    epoch     = 0;

    while epoch < numEpochs && ~monitor.Stop
        epoch = epoch + 1;

        % -----------------------------------------------------------------
        % Multi-resolution schedule: switch to full-resolution at the
        % configured epoch threshold. setLevel() swaps obj.data.images
        % from imagesHalf to imagesFull without reloading any data.
        % -----------------------------------------------------------------
        if epoch == levelSwitchEpoch
            obj.data.setLevel(1);
            % Rebuild combined datastore and minibatchqueue at new resolution
            ds  = combine(obj.data.images, obj.data.cameras);
            mbq = minibatchqueue(ds, ...
                'MiniBatchSize',    miniBatchSize, ...
                'PartialMiniBatch', 'discard', ...
                'OutputEnvironment','gpu', ...
                'MiniBatchFormat',  ["SSCB","CB","CB","CB","CB","CB","CB","CB","SSCB","SCB","SCB"]);
            fprintf('Switched to full-resolution training at epoch %d.\n', epoch);
        end

        % Shuffle data at the start of each epoch
        shuffle(mbq);

        while hasdata(mbq) && ~monitor.Stop
            iteration = iteration + 1;

            % Fetch next mini-batch of image tiles and camera parameters.
            % Data arrives on GPU due to OutputEnvironment='gpu'.
            % Camera scalars come as 'CB' format — squeeze removes the
            % leading channel dimension to give plain 1×B vectors.
            [obj.image_gt, camId, camW, camH, camFx, camFy, camCx, camCy, ...
             obj.camera.Rcw, obj.camera.tcw, obj.camera.twc] = next(mbq);

            obj.camera.id     = squeeze(camId);
            obj.camera.width  = squeeze(camW);
            obj.camera.height = squeeze(camH);
            obj.camera.fx     = squeeze(camFx);
            obj.camera.fy     = squeeze(camFy);
            obj.camera.cx     = squeeze(camCx);
            obj.camera.cy     = squeeze(camCy);

            if iteration == 1
                % Initialize GPU storage once on the first iteration
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
    [obj.image_gt, camId, camW, camH, camFx, camFy, camCx, camCy, ...
     obj.camera.Rcw, obj.camera.tcw, obj.camera.twc] = next(mbq);

    obj.camera.id     = squeeze(camId);
    obj.camera.width  = squeeze(camW);
    obj.camera.height = squeeze(camH);
    obj.camera.fx     = squeeze(camFx);
    obj.camera.fy     = squeeze(camFy);
    obj.camera.cx     = squeeze(camCx);
    obj.camera.cy     = squeeze(camCy);

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