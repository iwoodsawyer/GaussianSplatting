%% View Gaussian Splats as a 3D Point Cloud
% This script densifies the Gaussian Splats in batches and merges them
% into a single point cloud. The approach is memory-efficient and prevents
% over-drawing using grid-based downsampling (pcdownsample).
%
% Parameter decoding mirrors the latest GaussianSplatter:
%   - Opacity:  sigmoid of alphas_raw, clamped to <= 1
%   - Scale:    exp of scales_raw clamped to [-10, 10]
%   - Rotation: normalized quaternion (Rodrigues rotation, same as R*diag(s))
%   - Color:    per-channel second-order SH evaluation on shs [N x 9 x 3]
%               (computeColors convention; view direction = normalized position)
%
% Dependencies: gaussians.mat produced by trainGaussianSplat.m

clear; clc;

% --- Configuration ---
filename       = 'gaussians.mat';
pointsPerSplat = 1e6;    % Points to generate per Gaussian (controls density)
gridStep       = 0.1;    % Grid cell size for pcdownsample (visual resolution)
alphaThresh    = 0.018;  % ~ sigmoid(forcePruneThreshold = -4): near-invisible
batchSize      = 5000;   % Gaussians processed per batch (bounds peak memory)

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

% Scale: exponentiate clamped log-domain values
scale = exp(min(max(scales_raw, single(-10)), single(10)));

% Normalize quaternions to unit length
quat = rots_raw ./ max(vecnorm(rots_raw, 2, 2), 1e-6);

% Spherical harmonic colors, per channel (computeColors convention).
% View direction = unit-normalized world position (no camera available).
vd = pws ./ max(vecnorm(pws, 2, 2), 1e-6);
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

%% 3. Batched Densification
% For each batch of Gaussians: sample points with Latin Hypercube Sampling,
% rotate by the stored quaternion, translate to world space, and downsample.
numBatches = ceil(N / batchSize);
ptClouds   = repmat(pointCloud(zeros(0,3,'single')), numBatches, 1);
fprintf('Processing %d batches...\n', numBatches);

% Number of sample points proportional to volume x opacity
nPtsPer = ceil(prod(scale, 2) .* alpha .* pointsPerSplat);

for b = 1:numBatches
    idx  = (b-1)*batchSize + 1 : min(b*batchSize, N);
    nPts = nPtsPer(idx);
    K    = sum(nPts);
    gidx = repelem(idx(:), nPts);   % per-point Gaussian index

    % Latin Hypercube Sampling mapped to Gaussian sphere via inverse normal CDF
    pts = single(norminv(lhsdesign(K, 3, 'criterion', 'none')));

    % Scale, then rotate by quaternion (Rodrigues; equals R*diag(s) in splatter)
    pts   = pts .* scale(gidx, :);
    q_w   = quat(gidx, 1);
    q_vec = quat(gidx, 2:4);
    t     = 2 * cross(q_vec, pts, 2);
    pts   = pts + (q_w .* t) + cross(q_vec, t, 2);

    % Translate to world-space Gaussian centres
    pts = pts + pws(gidx, :);

    % Downsample per batch to bound memory before the final merge
    ptClouds(b) = pcdownsample(pointCloud(pts, 'Color', colors(gidx, :)), ...
        "gridAverage", gridStep);
end

%% 4. Merge, Downsample, and Display
% pccat concatenates all per-batch point clouds into one.
% A final pcdownsample removes redundant points across batch boundaries.
ptCloud = pccat(ptClouds);
ptCloud = pcdownsample(ptCloud, "gridAverage", gridStep);
fprintf('Final point cloud size: %d points\n', ptCloud.Count);

% Use pcviewer (Image Processing Toolbox) if available, otherwise pcshow.
if exist('pcviewer', 'file') == 2
    pcviewer(ptCloud);
else
    figure('Name', 'Gaussian Splat Point Cloud');
    pcshow(ptCloud, 'VerticalAxis', 'y', 'VerticalAxisDir', 'down');
    title('Merged Gaussian Splat Result');
end