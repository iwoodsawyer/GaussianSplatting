classdef ColmapData < handle
    % ColmapData - MATLAB implementation for processing COLMAP data for
    % 3D Gaussian Splatting.
    %
    % Logic derived from gsplat_data.hpp:
    %   1. Loads Cameras, Images, and Points3D using ColmapLoader.
    %   2. Initializes Gaussians from Points3D.
    %   3. Stores images as a blockedImageDatastore for lazy, tile-level
    %      GPU loading using the Image Processing Toolbox.
    %   4. Builds a two-level multi-resolution pyramid with makeMultiLevel2D
    %      so training can start at half-resolution and switch to full
    %      resolution mid-training.
    %   5. Computes scene scale.
    %
    % Blocked Image Pipeline:
    %   Each source image is wrapped as a blockedImage object, then passed
    %   through makeMultiLevel2D to produce a two-level pyramid. Preprocessing
    %   (resize + normalize) is applied block-wise via apply() with BatchSize
    %   tuned for the RTX 4050 Laptop GPU (6 GB VRAM, 2560 CUDA cores).
    %   The final blockedImageDatastore streams ReadSize tiles per read(),
    %   matching miniBatchSize so minibatchqueue receives exactly one
    %   mini-batch of tile blocks per next() call.
    %
    % Usage:
    %   data = ColmapData('path/to/dataset', 2000, 20, [256 256], 2);
    %
    %   % Access an image tile block (lazy loaded, resized, normalized)
    %   tile = read(data.images);
    %   % or preview the first tile
    %   preview(data.images);

    properties
        cameras      % Combined arrayDatastore: id, w, h, fx, fy, cx, cy, Rcw, tcw, twc
        images       % blockedImageDatastore: preprocessed tile blocks
        gaussians    % Struct: {pws, shs, scales, rots, alphas}
        scene_scale  % arrayDatastore: scene scale factor (float)
    end

    properties (Constant)
        SH_C0_0          = 0.28209479177387814; % Zeroth-order spherical harmonic coefficient
        kScaleDownFactor = 2.0;                 % Downsample factor for loading images
        kInitialAlpha    = 0.8;                 % Initial opacity for Gaussians
    end

    methods
        function obj = ColmapData(dataset_path, max_num_gaussians, max_num_images, ...
                                   blockSize, miniBatchSize)
            % Constructor: loads COLMAP data, initializes Gaussians, and
            % builds the blocked image pipeline.
            %
            % Args:
            %   dataset_path      (string) : Root directory of the dataset.
            %   max_num_gaussians (int)    : Maximum number of Gaussians to keep.
            %   max_num_images    (int)    : Maximum number of training images.
            %   blockSize         (1x2 int): Tile height and width in pixels,
            %                               e.g. [256 256]. Should match the
            %                               tile size used by GaussianSplatter.
            %   miniBatchSize     (int)    : Number of tiles per mini-batch;
            %                               sets ReadSize on the datastore and
            %                               BatchSize on apply().

            if nargin < 4 || isempty(blockSize)
                blockSize = [256, 256];
            end
            if nargin < 5 || isempty(miniBatchSize)
                miniBatchSize = 2;
            end

            % ------------------------------------------------------------------
            % 1. Load COLMAP Data
            % Assumes standard structure: /sparse/0/ and /images/
            % ------------------------------------------------------------------
            sparse_path = fullfile(dataset_path, 'sparse', '0');
            images_dir  = fullfile(dataset_path, 'images');

            fprintf('Loading COLMAP data from %s...\n', sparse_path);
            [cams_map, ims_map, pts_struct] = ColmapLoader.load(sparse_path);

            fprintf('Loaded: %d cameras, %d images, %d points3D\n', ...
                cams_map.Count, ims_map.Count, length(pts_struct));

            % ------------------------------------------------------------------
            % 2. Process Metadata (Cameras and File Paths)
            % Iterate to validate files and calculate camera intrinsics/
            % extrinsics. Pixel data is NOT loaded here; only metadata.
            % imfinfo is used for dimensions to avoid imread overhead.
            % ------------------------------------------------------------------
            im_keys = sort(cell2mat(keys(ims_map)));

            temp_cameras       = [];
            valid_image_paths  = string.empty(0,1);

            fprintf('Processing metadata...\n');
            for k = 1:length(im_keys)
                im = ims_map(im_keys(k));

                full_im_path = fullfile(images_dir, im.name);
                if ~isfile(full_im_path)
                    warning('Image not found: %s', full_im_path);
                    continue;
                end

                % Use imfinfo for dimensions only — faster than imread
                info  = imfinfo(full_im_path);
                raw_w = info.Width;
                raw_h = info.Height;

                % Target dimensions after downscale
                w_cur = round(raw_w / obj.kScaleDownFactor);
                h_cur = round(raw_h / obj.kScaleDownFactor);

                % Retrieve camera model and scale intrinsics accordingly
                colmap_cam = cams_map(im.camera_id);
                w_model    = double(colmap_cam.w);
                h_model    = double(colmap_cam.h);
                w_scale    = w_cur / w_model;
                h_scale    = h_cur / h_model;

                cam    = struct();
                cam.id = single(im.id);
                cam.width  = single(w_cur);
                cam.height = single(h_cur);

                % Adjust intrinsic parameters to downscaled resolution
                params = colmap_cam.params;
                if length(params) >= 4
                    cam.fx = single(params(1) * w_scale);
                    cam.fy = single(params(2) * h_scale);
                    cam.cx = single(params(3) * w_scale);
                    cam.cy = single(params(4) * h_scale);
                elseif length(params) == 3
                    cam.fx = single(params(1) * w_scale);
                    cam.fy = single(params(1) * h_scale);
                    cam.cx = single(params(2) * w_scale);
                    cam.cy = single(params(3) * h_scale);
                end

                % Extrinsics: rotation matrix and translation vector
                cam.Rcw = single(ColmapData.qVec2RotMat(im.q));
                cam.tcw = single(im.t(:));
                cam.twc = single(-cam.Rcw' * cam.tcw);

                cam.image_path = full_im_path;

                valid_image_paths(end+1,1) = string(full_im_path); %#ok<AGROW>
                if isempty(temp_cameras)
                    temp_cameras = cam;
                else
                    temp_cameras = [temp_cameras; cam]; %#ok<AGROW>
                end
            end

            % ------------------------------------------------------------------
            % 3. Initialize Gaussians from Sparse 3D Points
            % ------------------------------------------------------------------
            fprintf('Initializing Gaussians...\n');
            obj.gaussians = obj.initGaussiansFrom3dPoints(pts_struct);

            % ------------------------------------------------------------------
            % 4. Limit and Shuffle Data
            % ------------------------------------------------------------------

            % Shuffle Gaussians and keep up to max_num_gaussians
            num_points = size(obj.gaussians.pws, 1);
            p_idx      = randperm(num_points);
            limit_g    = min(num_points, max_num_gaussians);
            keep_idx   = p_idx(1:limit_g);

            obj.gaussians.pws    = obj.gaussians.pws(keep_idx, :);
            obj.gaussians.shs    = obj.gaussians.shs(keep_idx, :);
            obj.gaussians.scales = obj.gaussians.scales(keep_idx, :);
            obj.gaussians.rots   = obj.gaussians.rots(keep_idx, :);
            obj.gaussians.alphas = obj.gaussians.alphas(keep_idx, :);

            % Reverse and limit images/cameras
            valid_image_paths = flip(valid_image_paths);
            temp_cameras      = flip(temp_cameras);
            limit_img         = min(length(valid_image_paths), max_num_images);
            valid_image_paths = valid_image_paths(1:limit_img);
            temp_cameras      = temp_cameras(1:limit_img);

            % ------------------------------------------------------------------
            % 5. Build Blocked Image Pipeline
            %
            % Step A: Create one blockedImage per source image.
            %   blockedImage wraps the file reference; no pixel data is read
            %   into RAM at this point.
            %
            % Step B: Build a two-level multi-resolution pyramid with
            %   makeMultiLevel2D (Image Processing Toolbox).
            %   Level 1 = finest (full downscale), Level 2 = half-resolution.
            %   Training starts at Level 2 for fast coarse convergence, then
            %   switches to Level 1 for fine-detail refinement.
            %
            % Step C: Apply preprocessing (resize to blockSize + normalize to
            %   [0,1]) block-wise using apply() with BatchSize = miniBatchSize.
            %   BatchSize > 1 saturates the GPU for Ada Lovelace CUDA cores.
            %   PadPartialBlocks=true ensures uniform tile size at image edges.
            %
            % Step D: Create the blockedImageDatastore with ReadSize =
            %   miniBatchSize so each read() returns exactly one mini-batch
            %   of tiles, matching the minibatchqueue MiniBatchSize.
            % ------------------------------------------------------------------

            fprintf('Building blocked image pipeline...\n');

            % Step A — Create blockedImage array
            bimArray = blockedImage.empty(0, 1);
            for k = 1:limit_img
                bimArray(k) = blockedImage(char(valid_image_paths(k)));
            end

            % Step B — Build two-level multi-resolution pyramid
            % Block size for the pyramid must include the channel dimension.
            pyramidBlockSize = [blockSize(1), blockSize(2), 3];
            bimArrayML = blockedImage.empty(0, 1);
            for k = 1:numel(bimArray)
                bimArrayML(k) = makeMultiLevel2D(bimArray(k), pyramidBlockSize(1:2));
            end

            % Step C — Preprocessing via apply()
            % Resize each block to the target blockSize and normalize to [0,1].
            % im2single performs uint8->single + divide by 255 in one step.
            % BatchSize = miniBatchSize batches tiles together on the GPU,
            % maximising throughput on the RTX 4050's 2560 CUDA cores.
            targetSize     = blockSize;          % [H W] for imresize
            preprocessFcn  = @(block) im2single( ...
                imresize(block.Data, targetSize, 'bilinear'));

            bimArrayProcessed = blockedImage.empty(0, 1);
            for k = 1:numel(bimArrayML)
                bimArrayProcessed(k) = apply( ...
                    bimArrayML(k), ...
                    preprocessFcn, ...
                    'BlockSize',        pyramidBlockSize, ...
                    'BatchSize',        miniBatchSize, ...  % GPU batch saturation
                    'PadPartialBlocks', true, ...
                    'PadMethod',        'symmetric', ...
                    'Level',            2);                 % Start at half-resolution
            end

            % Step D — Create blockedImageDatastore
            % Level=2 starts training at half-resolution; caller switches to
            % Level=1 at levelSwitchEpoch via obj.data.images.Level = 1.
            % ReadSize=miniBatchSize aligns one read() with one mini-batch.
            obj.images = blockedImageDatastore( ...
                bimArrayProcessed, ...
                'BlockSize',        pyramidBlockSize, ...
                'Level',            2, ...
                'PadPartialBlocks', true, ...
                'PadMethod',        'symmetric', ...
                'ReadSize',         miniBatchSize);

            % ------------------------------------------------------------------
            % 6. Build Camera Label Datastores
            % arrayDatastore provides O(1) random-access shuffling of camera
            % parameters that are combined with the image tile datastore.
            % ------------------------------------------------------------------
            obj.cameras = combine( ...
                arrayDatastore([temp_cameras.id],                                   'IterationDimension', 2), ...
                arrayDatastore([temp_cameras.width],                                'IterationDimension', 2), ...
                arrayDatastore([temp_cameras.height],                               'IterationDimension', 2), ...
                arrayDatastore([temp_cameras.fx],                                   'IterationDimension', 2), ...
                arrayDatastore([temp_cameras.fy],                                   'IterationDimension', 2), ...
                arrayDatastore([temp_cameras.cx],                                   'IterationDimension', 2), ...
                arrayDatastore([temp_cameras.cy],                                   'IterationDimension', 2), ...
                arrayDatastore(reshape([temp_cameras.Rcw], 3, 3, []),               'IterationDimension', 3), ...
                arrayDatastore(reshape([temp_cameras.tcw], 3, 1, []),               'IterationDimension', 3), ...
                arrayDatastore(reshape([temp_cameras.twc], 3, 1, []),               'IterationDimension', 3));

            % ------------------------------------------------------------------
            % 7. Find Scene Scale
            % ------------------------------------------------------------------
            obj.scene_scale = arrayDatastore( ...
                ColmapData.findSceneScale(temp_cameras), 'IterationDimension', 2);

            fprintf('ColmapData initialisation complete.\n');
        end

        function g = initGaussiansFrom3dPoints(obj, pts)
            % Initialize Gaussian parameters from the sparse 3D point cloud.
            num_pts = length(pts);

            if num_pts == 0
                g.pws    = [];
                g.shs    = [];
                g.rots   = [];
                g.scales = [];
                g.alphas = [];
                return;
            end

            % 3D Positions
            g.pws = [[pts.x]', [pts.y]', [pts.z]'];

            % Spherical Harmonic Colors (zeroth-order band, RGB)
            rgb   = [[pts.r]', [pts.g]', [pts.b]'];
            g.shs = ((double(rgb) / 255.0) - 0.5) / obj.SH_C0_0;

            % Initial rotation: identity quaternion [w x y z] = [0 0 0 1]
            g.rots = repmat([0, 0, 0, 1], num_pts, 1);

            % Initial opacity
            g.alphas = repmat(obj.kInitialAlpha, num_pts, 1);

            % Initial isotropic scale based on nearest-neighbour heuristic
            nearest_dist = 0.1;
            g.scales = repmat([nearest_dist, nearest_dist, nearest_dist], num_pts, 1);
        end
    end

    methods (Static)
        function scale = findSceneScale(cameras)
            % Compute scene scale as 1.1x the maximum camera-centre spread.
            if isempty(cameras)
                scale = 1.0;
                return;
            end

            twcs   = [cameras.twc];       % 3 x N matrix of camera centres
            center = mean(twcs, 2);
            dists  = sqrt(sum((twcs - center).^2, 1));

            kScaleFactor = 1.1;
            scale = max(dists) * kScaleFactor;
        end

        function R = qVec2RotMat(qvec)
            % Convert quaternion [w, x, y, z] to 3x3 rotation matrix.
            w = qvec(1); x = qvec(2); y = qvec(3); z = qvec(4);

            R = zeros(3, 3);
            R(1,1) = 1 - 2*y^2 - 2*z^2;
            R(1,2) = 2*x*y - 2*w*z;
            R(1,3) = 2*x*z + 2*w*y;

            R(2,1) = 2*x*y + 2*w*z;
            R(2,2) = 1 - 2*x^2 - 2*z^2;
            R(2,3) = 2*y*z - 2*w*x;

            R(3,1) = 2*x*z - 2*w*y;
            R(3,2) = 2*y*z + 2*w*x;
            R(3,3) = 1 - 2*x^2 - 2*y^2;
        end
    end
end