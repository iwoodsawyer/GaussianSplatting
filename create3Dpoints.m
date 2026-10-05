%% View Gaussian Splats as a 3D Point Cloud
% Inspect Gaussian centres without filling the scene with background splats.
% Optional sampled mode densifies in batches and downsamples the result.
%
% Parameter decoding mirrors the latest GaussianSplatter:
%   - Opacity:  sigmoid of alphas_raw, clamped to <= 1
%   - Scale:    exp of scales_raw clamped to [-10, 10]
%   - Rotation: normalized quaternion, matching the renderer's matrix layout
%   - Color:    per-channel second-order SH evaluation on shs [N x 9 x 3]
%               or view-independent DC color (default for inspection)
%
% Dependencies: gaussians.mat produced by trainGaussianSplat.m

clear; clc;

% --- Configuration ---
filename       = 'gaussians.mat';
cloudMode      = "sampled"; % "centres" or "sampled"
colorMode      = "dc";   % "dc" or "sh" (view-dependent)
cameraPosition = [0 0 0]; % World-space camera used only for SH colors
datasetPath    = 'C:\Source\tandt_db\tandt\train'; % Dataset root containing sparse/0/images.bin
imageId        = 1;     % Optional COLMAP image ID; [] keeps cameraPosition
cameraRotation = [];     % COLMAP world-to-camera rotation; set when imageId is used
focusPercentiles = [5 95]; % Initial view bounds; [] shows the full extent
pointsPerSplat = 1e5;    % Sample density per volume x opacity (sampled mode)
gridStep       = 0.1;    % Grid cell size for pcdownsample (visual resolution)
alphaThresh    = 0.018;  % ~ sigmoid(forcePruneThreshold = -4): near-invisible
batchSize      = 5000;   % Gaussians processed per batch (bounds peak memory)

cloudMode = validatestring(cloudMode, {'centres', 'sampled'});
colorMode = validatestring(colorMode, {'dc', 'sh'});
if ~isempty(imageId)
    validateattributes(imageId, {'numeric'}, ...
        {'real', 'finite', 'scalar', 'integer', 'positive', '<=', double(intmax('int32'))});
    if isempty(datasetPath)
        error('create3Dpoints:MissingDatasetPath', ...
            'Set datasetPath to the dataset root when imageId is specified.');
    end
    imagesFile = fullfile(datasetPath, 'sparse', '0', 'images.bin');
    if ~isfile(imagesFile)
        error('create3Dpoints:MissingImagesFile', ...
            'COLMAP images.bin not found: %s', imagesFile);
    end
    images = ColmapLoader.loadImages(imagesFile);
    if ~isKey(images, int32(imageId))
        error('create3Dpoints:UnknownImageId', ...
            'COLMAP image ID %d was not found in %s.', imageId, imagesFile);
    end
    image = images(int32(imageId));
    Rcw = ColmapData.qVec2RotMat(image.q);
    cameraRotation = Rcw;
    cameraPosition = (-Rcw' * image.t(:))';
    fprintf('Camera centre for COLMAP image ID %d (%s): [%g %g %g]\n', ...
        imageId, image.name, cameraPosition);
end
validateattributes(cameraPosition, {'numeric'}, {'real', 'finite', 'size', [1 3]});
if ~isempty(focusPercentiles)
    validateattributes(focusPercentiles, {'numeric'}, ...
        {'real', 'finite', 'size', [1 2], '>=', 0, '<=', 100});
    assert(focusPercentiles(1) < focusPercentiles(2), ...
        'Focus percentiles must be strictly increasing.');
end

%% 1. Load Trained Gaussian Parameters
if ~isfile(filename)
    error('gaussians.mat not found. Run trainGaussianSplat.m first.');
end
data = load(filename, 'params');

% Helper: extract from dlarray/gpuArray to CPU single
ext = @(x) single(gather(extractdata(x)));

pws        = ext(data.params.pws);         % [N x 3]
shs        = ext(data.params.shs);         % [N x 9 x 3]
scales_raw = ext(data.params.scales_raw);  % [N x 3]
rots_raw   = ext(data.params.rots_raw);    % [N x 4]
alphas_raw = ext(data.params.alphas_raw);  % [N x 1]

% Zeroth-order and second-order spherical harmonic basis coefficients
shToColor = single([0.28209479177387814; ...
                    0.4886025119029199;  ...
                    0.4886025119029199;  ...
                    0.4886025119029199;  ...
                    1.0925484305920792;  ...
                    1.0925484305920792;  ...
                    1.0925484305920792;  ...
                    0.31539156525252005; ...
                    0.5462742152960396]);

totalGaussians = size(pws, 1);
fprintf('Total Gaussians loaded: %d\n', totalGaussians);

%% 2. Decode Parameters (vectorized, matching GaussianSplatter)
% Opacity: clamped sigmoid of raw logit
alpha = min(single(1.0) ./ (single(1.0) + exp(-alphas_raw)), single(1.0));

% Cull near-transparent Gaussians
keep = alpha >= alphaThresh;
pws        = pws(keep, :);
shs        = shs(keep, :, :);
scales_raw = scales_raw(keep, :);
rots_raw   = rots_raw(keep, :);
alpha      = alpha(keep, :);
N = size(pws, 1);
fprintf('Gaussians after opacity culling: %d\n', N);
if N == 0
    error('create3Dpoints:EmptyCloud', ...
        'No Gaussians pass alphaThresh. Lower the threshold or check the saved parameters.');
end

% Scale: exponentiate clamped log-domain values
scale = exp(min(max(scales_raw, single(-10)), single(10)));

% Normalize quaternions to unit length
quat = rots_raw ./ max(vecnorm(rots_raw, 2, 2), 1e-6);

% Spherical harmonic colors, per channel (computeColors convention).
% SH colors are evaluated once for the configured camera, not the viewer.
vd = pws - single(cameraPosition);
vd = vd ./ max(vecnorm(vd, 2, 2), 1e-6);
vx = vd(:,1); vy = vd(:,2); vz = vd(:,3);

c  = shToColor;
Sh = [c(1) .* ones(N, 1, 'single'), ...
      c(2) .* (-vx), ...
      c(3) .* (-vy), ...
      c(4) .* vz, ...
      c(5) .* (vx .* vy), ...
      c(6) .* (-vx .* vz), ...
      c(7) .* (-vy .* vz), ...
      c(8) .* (single(3.0) .* vz .* vz - single(1.0)), ...
      c(9) .* (vx .* vx - vy .* vy)];                     % [N x 9]

colR = max(min(single(0.5) + sum(Sh .* shs(:,:,1), 2), single(1.0)), single(0.0));
colG = max(min(single(0.5) + sum(Sh .* shs(:,:,2), 2), single(1.0)), single(0.0));
colB = max(min(single(0.5) + sum(Sh .* shs(:,:,3), 2), single(1.0)), single(0.0));
colors = [colR, colG, colB];                              % [N x 3]
if strcmp(colorMode, 'dc')
    colors = max(min(single(0.5) + c(1) .* reshape(shs(:,1,:), N, 3), ...
        single(1.0)), single(0.0));
end

%% 3. Build Point Cloud
if strcmp(cloudMode, 'centres')
    % Preserve small splats and their colors: no sampling or downsampling.
    ptCloud = pointCloud(pws, 'Color', colors);
else
    numBatches = ceil(N / batchSize);
    ptClouds   = repmat(pointCloud(zeros(0,3,'single')), numBatches, 1);
    fprintf('Processing %d batches...\n', numBatches);

    % Sum volume-weighted counts in double precision to match repeated indices.
    nPtsPer = double(ceil(prod(scale, 2) .* alpha .* pointsPerSplat));

    for b = 1:numBatches
        idx  = (b-1)*batchSize + 1 : min(b*batchSize, N);
        nPts = nPtsPer(idx);
        K    = sum(nPts);
        gidx = repelem(idx(:), nPts);   % per-point Gaussian index

        % Latin Hypercube Sampling mapped via inverse normal CDF
        pts = single(norminv(lhsdesign(K, 3, 'criterion', 'none')));

        % R_cols is reshaped column-wise in the renderer, giving the conjugate
        % quaternion rotation. Match that layout for anisotropic sample axes.
        pts   = pts .* scale(gidx, :);
        q_w   = quat(gidx, 1);
        q_vec = -quat(gidx, 2:4);
        t     = 2 * cross(q_vec, pts, 2);
        pts   = pts + (q_w .* t) + cross(q_vec, t, 2);
        pts   = pts + pws(gidx, :);

        ptClouds(b) = pcdownsample(pointCloud(pts, 'Color', colors(gidx, :)), ...
            "gridAverage", gridStep);
    end

    % Remove redundant points across batch boundaries.
    ptCloud = pcdownsample(pccat(ptClouds), "gridAverage", gridStep);
end
fprintf('Final point cloud size: %d points\n', ptCloud.Count);

%% 4. Display
% Display with controllable axis limits; retain all points in ptCloud.
figure('Name', 'Gaussian Splat Point Cloud');
ax = pcshow(ptCloud, 'VerticalAxis', 'y', 'VerticalAxisDir', 'down');
title(ax, sprintf('Gaussian Splat Point Cloud (%s, %s colors)', cloudMode, colorMode));
if ~isempty(focusPercentiles)
    bounds = double(prctile(pws, focusPercentiles, 1));
    padding = max(0.05 .* (bounds(2,:) - bounds(1,:)), 1e-3);
    bounds = bounds + [-padding; padding];
    xlim(ax, bounds(:,1)');
    ylim(ax, bounds(:,2)');
    zlim(ax, bounds(:,3)');
    fprintf('Initial view focused on centre percentiles [%g %g]; full cloud retained in ptCloud.\n', ...
        focusPercentiles);
else
    bounds = [min(double(pws), [], 1); max(double(pws), [], 1)];
end
if ~isempty(imageId)
    viewDistance = max(norm(bounds(2,:) - bounds(1,:)), 1);
    cameraForward = (cameraRotation' * [0; 0; 1])';
    cameraUp = (cameraRotation' * [0; -1; 0])';
    campos(ax, cameraPosition);
    camtarget(ax, cameraPosition + cameraForward .* viewDistance);
    camup(ax, cameraUp);
end