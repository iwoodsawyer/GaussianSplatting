%% View Gaussian Splats as a 3D Point Cloud
% This script densifies the Gaussian Splats in batches and merges them
% into a single point cloud. The approach is memory-efficient and prevents
% over-drawing using grid-based downsampling (pcdownsample).
%
% Dependencies: gaussians.mat produced by trainGaussianSplat.m

clear; clc;

% --- Configuration ---
filename       = 'gaussians.mat';
pointsPerSplat = 1e6;   % Points to generate per Gaussian (controls density)
gridStep       = 0.1;   % Grid cell size for pcdownsample (visual resolution)
alphaThresh    = 0.01;  % Minimum opacity — ignore near-transparent Gaussians

%% 1. Load Trained Gaussian Parameters
if ~isfile(filename)
    error('gaussians.mat not found. Run trainGaussianSplat.m first.');
end
data = load(filename, 'params');

% Helper: extract from dlarray/gpuArray to CPU double/single
ext = @(x) gather(extractdata(x));

pws    = ext(data.params.pws);
shs    = ext(data.params.shs);
scales = ext(data.params.scales_raw);
rots   = ext(data.params.rots_raw);
alphas = ext(data.params.alphas_raw);

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

%% 2. Preallocate Point Cloud Array
totalGaussians = size(pws, 1);
ptCloud = repmat(pointCloud(zeros(0,3), 'Color', zeros(0,3)), totalGaussians, 1);
fprintf('Total Gaussians loaded: %d\n', totalGaussians);

%% 3. Densification Loop
% For each Gaussian: evaluate opacity, sample points using Latin Hypercube
% Sampling, rotate by the stored quaternion, and compute RGB from SH coefficients.
fprintf('Processing and merging Gaussians...\n');

for i = 1:totalGaussians

    % Extract per-Gaussian parameters
    pw        = pws(i,:);
    sh        = shs(i,:,:);
    scale_raw = scales(i,:);
    rot_raw   = rots(i,:);
    alpha_raw = alphas(i,:);

    % Opacity: sigmoid of raw logit
    alpha = 1 ./ (1 + exp(-alpha_raw));

    % Skip near-transparent Gaussians (culling)
    if all(alpha < alphaThresh)
        continue;
    end

    % Scale: exponentiate from log domain
    scale = exp(scale_raw);

    % Number of sample points proportional to volume × opacity
    nr_pts = ceil(prod(scale) .* alpha .* pointsPerSplat);

    % Latin Hypercube Sampling: generates well-distributed unit-cube samples
    pts = single(lhsdesign(nr_pts, 3, 'criterion', 'correlation'));

    % Map to Gaussian sphere via inverse normal CDF
    pts = norminv(pts);
    pts = pts .* scale;

    % Normalize quaternion to unit length
    quat  = rot_raw;
    quat  = quat ./ max(vecnorm(quat), 1e-6);

    % Rotate sample points by the Gaussian's quaternion (Rodrigues' formula)
    q_w   = ones(nr_pts, 1) * quat(:,1);
    q_vec = ones(nr_pts, 1) * quat(:,2:4);
    t     = 2 * cross(q_vec, pts, 2);
    pts   = pts + (q_w .* t) + cross(q_vec, t, 2);

    % Translate to world-space Gaussian centre
    pts = pw + pts;

    % Unit-normalise position for spherical harmonic colour evaluation
    pw_norm = pw ./ max(vecnorm(pw, 2, 2), 1e-6);

    % Evaluate second-order spherical harmonic basis at normalised position
    Sh = [shToColor(1), ...
          shToColor(2) .* (-pw_norm(:,1)), ...
          shToColor(3) .* (-pw_norm(:,2)), ...
          shToColor(4) .* ( pw_norm(:,3)), ...
          shToColor(5) .* ( pw_norm(:,1) .* pw_norm(:,2)), ...
          shToColor(6) .* (-pw_norm(:,1) .* pw_norm(:,3)), ...
          shToColor(7) .* (-pw_norm(:,2) .* pw_norm(:,3)), ...
          shToColor(8) .* (single(3.0) .* pw_norm(:,3).^2 - single(1.0)), ...
          shToColor(9) .* (pw_norm(:,1).^2 - pw_norm(:,2).^2)];

    % RGB color: clamp SH expansion to [0, 1]
    colors = reshape(max(min(0.5 + sum(Sh .* sh, 2), 1), 0), 1, 3);

    % Build per-Gaussian point cloud
    ptCloud(i) = pointCloud(pts, 'Color', colors);
end

%% 4. Merge, Downsample, and Display
% pccat concatenates all per-Gaussian point clouds into one.
% pcdownsample with 'gridAverage' removes redundant points within each
% grid cell, keeping the display memory-efficient.
ptCloud = pccat(ptCloud);
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