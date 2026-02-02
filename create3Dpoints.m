%% View Gaussian Splats with pcmerge
% This script densifies the Gaussian Splats in batches and merges them
% into a single point cloud directly. This approach is memory efficient
% and prevents over-drawing using grid-based filtering.

clear; clc;

% --- Configuration ---
filename = 'gaussians.mat';
pointsPerSplat = 1e6;     % Points to generate per Gaussian
gridStep       = 0.1;     % Distance to merge nearby points (Visual Resolution)
alphaThresh    = 0.01;    % Ignore transparent splats

%% 1. Load Data
if ~isfile(filename), error('gaussians.mat not found'); end
data = load(filename, 'params');

% Helper for GPU/dlarray extraction
ext = @(x) gather(extractdata(x));

pws    = ext(data.params.pws);
shs    = ext(data.params.shs);
scales = ext(data.params.scales_raw);
rots   = ext(data.params.rots_raw);
alphas = ext(data.params.alphas_raw);

shToColor = single([0.28209479177387814;...
    0.4886025119029199;...
    0.4886025119029199;...
    0.4886025119029199;...
    1.0925484305920792;...
    1.0925484305920792;...
    1.0925484305920792;...
    0.31539156525252005;...
    0.5462742152960396]);


% Create an empty point cloud array
totalGaussians = size(pws, 1);
ptCloud = repmat(pointCloud(zeros(0,3), 'Color', zeros(0,3)),totalGaussians,1);
fprintf('Total Gaussians loaded: %d\n', totalGaussians);

%% 3. Batch Processing Loop
fprintf('Processing and Merging...\n');
for i = 1:totalGaussians
    % Extract Gaussian
    pw        = pws(i,:);
    sh        = shs(i,:,:);
    scale_raw = scales(i,:);
    rot_raw   = rots(i,:);
    alpha_raw = alphas(i,:);
    
    % Filter Invisible (Culling)
    alpha = 1 ./ (1 + exp(-alpha_raw));
    if all(alpha < alphaThresh)
        continue
    end

    % Scale
    scale = exp(scale_raw);

    % Densification 
    nr_pts = ceil(prod(scale).*alpha.*pointsPerSplat); 

    % Creates uniform samples on latin hypercube
    pts = single(lhsdesign(nr_pts,3,'criterion','correlation'));

    % Convert samples to Gaussian sphere
    pts = norminv(pts);
    pts = pts.*scale;
    
    % Normalize Quaternions
    quat = rot_raw;
    quat = quat./max(vecnorm(quat),1e-6);

    % Rotate (Quaternion)
    q_w = ones(nr_pts,1)*quat(:, 1); 
    q_vec = ones(nr_pts,1)*quat(:, 2:4);
    t = 2*cross(q_vec, pts, 2);
    pts = pts + (q_w.*t) + cross(q_vec, t, 2);
    pts = pw + pts;

    % Normalize positions
    pw = pw./max(vecnorm(pw, 2, 2),1e-6);

    % Apply spherical color coefficients
    Sh = [shToColor(1),...
        shToColor(2).*(-pw(:,1)),...
        shToColor(3).*(-pw(:,2)),...
        shToColor(4).*( pw(:,3)),...
        shToColor(5).*( pw(:,1).*pw(:,2)),...
        shToColor(6).*(-pw(:,1).*pw(:,3)),...
        shToColor(7).*(-pw(:,2).*pw(:,3)),...
        shToColor(8).*( single(3.0).*pw(:,3).*pw(:,3) - single(1.0)),...
        shToColor(9).*( pw(:,1).*pw(:,1) - pw(:,2).*pw(:,2))];

    % Constants
    colors = reshape(max(min(0.5 + sum(Sh.*sh,2), 1), 0),1,3);

    % Concatenate cloud
    ptCloud(i) = pointCloud(pts, 'Color', colors);
end

%% 4. View Result
ptCloud = pccat(ptCloud);
ptCloud = pcdownsample(ptCloud,"gridAverage",gridStep);
fprintf('Final Point Cloud Count: %d\n', ptCloud.Count);

if exist('pcviewer', 'file') == 2
    pcviewer(ptCloud);
else
    figure('Name', 'Merged Gaussian Splats');
    pcshow(ptCloud, 'VerticalAxis', 'y', 'VerticalAxisDir', 'down');
    title('Merged Result');
end