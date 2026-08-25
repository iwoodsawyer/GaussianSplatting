classdef ColmapData < handle
    % ColmapData - MATLAB implementation for processing COLMAP data for
    % 3D Gaussian Splatting.
    %
    % Logic derived from gsplat_data.hpp:
    %   1. Loads Cameras, Images, and Points3D using ColmapLoader.
    %   2. Initializes Gaussians from Points3D.
    %   3. Stores images as a blockedImageDatastore for lazy, tile-level
    %      GPU loading using the Image Processing Toolbox.
    %   4. Builds two separate single-level blockedImageDatastores:
    %        - imagesHalf : half-resolution  (fast early-epoch convergence)
    %        - imagesFull : full resolution  (fine-detail refinement)
    %      obj.images points to imagesHalf initially.
    %      Call obj.setLevel(1) to switch to imagesFull mid-training.
    %   5. Computes scene scale.
    %
    % Blocked Image Pipeline design notes:
    %
    %   apply() always outputs a SINGLE-level blockedImage regardless of
    %   whether the input is multi-level. Therefore makeMultiLevel2D is not
    %   used here — instead two independent preprocessing passes are run:
    %     Pass 1: apply at 'Level',1 from the file-backed bimArray  → full-res
    %     Pass 2: apply at 'Level',1 after an imresize halving step  → half-res
    %   Both outputs use 'Adapter', images.blocked.InMemory (no disk writes).
    %
    %   selectBlockLocations always operates on single-level images
    %   (Levels must be <= 1 for single-level blockedImage objects).
    %
    %   blockedImageDatastore 'Level' is NOT a valid argument.
    %   Resolution is controlled entirely via BlockLocationSet.
    %
    % Usage:
    %   data = ColmapData('path/to/dataset', 2000, 20, [256 256], 2);
    %   data.setLevel(1);   % switch to full resolution mid-training
    %   tile = read(data.images);

    properties
        cameras      % Combined arrayDatastore: id,w,h,fx,fy,cx,cy,Rcw,tcw,twc
        images       % blockedImageDatastore currently active (half or full res)
        gaussians    % Struct: {pws, shs, scales, rots, alphas}
        scene_scale  % arrayDatastore: scene scale factor (float)
    end

    properties (Access = private)
        % Two independent single-level datastores for resolution switching.
        % apply() always produces single-level output so two separate
        % preprocessing passes are used rather than makeMultiLevel2D.
        imagesHalf          % blockedImageDatastore at half resolution
        imagesFull          % blockedImageDatastore at full resolution
    end

    properties (Constant)
        SH_C0_0          = 0.28209479177387814; % Zeroth-order SH coefficient
        kScaleDownFactor = 2.0;                 % Downsample factor for image loading
        kInitialAlpha    = 0.8;                 % Initial Gaussian opacity
    end

    methods
        function obj = ColmapData(dataset_path, max_num_gaussians, max_num_images, ...
                                   blockSize, miniBatchSize)
            % Constructor: loads COLMAP data, initialises Gaussians, and
            % builds the blocked image pipeline.
            %
            % Args:
            %   dataset_path      (string) : Root directory of the dataset.
            %   max_num_gaussians (int)    : Maximum number of Gaussians to keep.
            %   max_num_images    (int)    : Maximum number of training images.
            %   blockSize         (1x2 int): Tile [H W] in pixels, e.g. [256 256].
            %                               Must match GaussianSplatter.blockSize.
            %   miniBatchSize     (int)    : Tiles per mini-batch; sets ReadSize on
            %                               the datastore and BatchSize on apply().

            if nargin < 4 || isempty(blockSize)
                blockSize = [256, 256];
            end
            if nargin < 5 || isempty(miniBatchSize)
                miniBatchSize = 2;
            end

            % Full tile size including channel dimension [H W C]
            pyramidBlockSize = [blockSize(1), blockSize(2), 3];

            % Half-resolution tile target for the coarse training phase
            halfBlockSize    = [blockSize(1), blockSize(2)];   % imresize target
            halfPyramidSize  = [blockSize(1), blockSize(2), 3];

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
            % Pixel data is NOT loaded here — only metadata via imfinfo.
            % ------------------------------------------------------------------
            im_keys = sort(cell2mat(keys(ims_map)));

            temp_cameras      = [];
            valid_image_paths = string.empty(0, 1);

            fprintf('Processing metadata...\n');
            for k = 1:length(im_keys)
                im = ims_map(im_keys(k));

                full_im_path = fullfile(images_dir, im.name);
                if ~isfile(full_im_path)
                    warning('Image not found: %s', full_im_path);
                    continue;
                end

                % imfinfo avoids a full imread just to get dimensions
                info  = imfinfo(full_im_path);
                raw_w = info.Width;
                raw_h = info.Height;

                % Target dimensions after the kScaleDownFactor downscale
                w_cur = round(raw_w / obj.kScaleDownFactor);
                h_cur = round(raw_h / obj.kScaleDownFactor);

                colmap_cam = cams_map(im.camera_id);
                w_model    = double(colmap_cam.w);
                h_model    = double(colmap_cam.h);
                w_scale    = w_cur / w_model;
                h_scale    = h_cur / h_model;

                cam        = struct();
                cam.id     = single(im.id);
                cam.width  = single(w_cur);
                cam.height = single(h_cur);

                % Adjust intrinsics to the downscaled resolution
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

                % Extrinsics
                cam.Rcw = single(ColmapData.qVec2RotMat(im.q));
                cam.tcw = single(im.t(:));
                cam.twc = single(-cam.Rcw' * cam.tcw);

                cam.image_path = full_im_path;

                valid_image_paths(end+1, 1) = string(full_im_path); %#ok<AGROW>
                if isempty(temp_cameras)
                    temp_cameras = cam;
                else
                    temp_cameras = [temp_cameras; cam]; %#ok<AGROW>
                end
            end

            % ------------------------------------------------------------------
            % 3. Initialise Gaussians from Sparse 3D Points
            % ------------------------------------------------------------------
            fprintf('Initializing Gaussians...\n');
            obj.gaussians = obj.initGaussiansFrom3dPoints(pts_struct);

            % ------------------------------------------------------------------
            % 4. Limit and Shuffle Data
            % ------------------------------------------------------------------
            num_points = size(obj.gaussians.pws, 1);
            p_idx      = randperm(num_points);
            limit_g    = min(num_points, max_num_gaussians);
            keep_idx   = p_idx(1:limit_g);

            obj.gaussians.pws    = obj.gaussians.pws(keep_idx, :);
            obj.gaussians.shs    = obj.gaussians.shs(keep_idx, :);
            obj.gaussians.scales = obj.gaussians.scales(keep_idx, :);
            obj.gaussians.rots   = obj.gaussians.rots(keep_idx, :);
            obj.gaussians.alphas = obj.gaussians.alphas(keep_idx, :);

            % Reverse and limit images/cameras (matches original C++ ordering)
            valid_image_paths = flip(valid_image_paths);
            temp_cameras      = flip(temp_cameras);
            limit_img         = min(length(valid_image_paths), max_num_images);
            valid_image_paths = valid_image_paths(1:limit_img);
            temp_cameras      = temp_cameras(1:limit_img);

            % ------------------------------------------------------------------
            % 5. Build Blocked Image Pipeline
            %
            % DESIGN: Two separate single-level preprocessed datastores.
            %
            % apply() always outputs a single-level blockedImage — multi-level
            % pyramid output is not preserved through apply(). Therefore the
            % resolution schedule is implemented as two independent preprocessing
            % passes, each producing a single-level InMemory blockedImage:
            %
            %   Pass A (full resolution):
            %     - Wrap source files as blockedImage objects.
            %     - apply(): resize each tile to blockSize + normalise to [0,1].
            %     - selectBlockLocations() at Level=1 (only valid level).
            %     - blockedImageDatastore() via BlockLocationSet.
            %
            %   Pass B (half resolution):
            %     - apply(): resize each tile to blockSize/2 + normalise.
            %     - selectBlockLocations() at Level=1.
            %     - blockedImageDatastore() via BlockLocationSet.
            %
            % obj.images starts pointing to imagesHalf (Pass B).
            % setLevel(1) swaps obj.images to imagesFull (Pass A).
            %
            % InMemory adapter is used for both passes — no disk writes,
            % no "file already exists" errors, no TIFF overhead.
            %
            % selectBlockLocations: 'Levels' must be <= NumLevels of the
            % input blockedImage. Since apply() output is always single-level,
            % 'Levels' must always be 1 here.
            % ------------------------------------------------------------------

            fprintf('Building blocked image pipeline...\n');

            % Step A1 — Create file-backed blockedImage objects (no pixel I/O)
            bimArray = blockedImage.empty(0, 1);
            for k = 1:limit_img
                bimArray(k) = blockedImage(char(valid_image_paths(k)), ...
                    'BlockSize', blockSize);    % [H W] — channel dim auto-appended
            end

            % ------------------------------------------------------------------
            % Pass A — Full-resolution preprocessing
            % Resize each tile to blockSize and normalise uint8 → single [0,1].
            % im2single handles uint8→single + /255 normalisation in one step.
            % BatchSize = miniBatchSize saturates the RTX 4050's 2560 CUDA cores.
            % InMemory adapter: no disk writes, no filename collision errors.
            % ------------------------------------------------------------------
            fullPreprocessFcn = @(block) im2single( ...
                imresize(block.Data, blockSize, 'bilinear'));

            bimFullArray = blockedImage.empty(0, 1);
            for k = 1:limit_img
                bimFullArray(k) = apply( ...
                    bimArray(k), ...
                    fullPreprocessFcn, ...
                    'BlockSize',        pyramidBlockSize, ...  % [H W C]
                    'BatchSize',        miniBatchSize, ...     % GPU tile batching
                    'PadPartialBlocks', true, ...
                    'PadMethod',        'symmetric', ...
                    'Adapter',          images.blocked.InMemory); % RAM only, no disk
            end

            % selectBlockLocations: apply() output is always single-level so
            % 'Levels' must be 1. This encodes the block grid into a
            % blockLocationSet for use by blockedImageDatastore.
            blsFull = selectBlockLocations(bimFullArray, ...
                'Levels',    1, ...                % single-level output from apply()
                'BlockSize', pyramidBlockSize);    % [H W C]

            % blockedImageDatastore.read() returns a cell array of blocks.
            % transform() unwraps each cell into a plain H×W×C single array so
            % that combine() can horzcat it cleanly with the numeric camera datastores.
            bimdsRaw = blockedImageDatastore(bimFullArray, ...
                'BlockLocationSet', blsFull, ...
                'PadPartialBlocks', true, ...
                'PadMethod',        'symmetric', ...
                'ReadSize',         miniBatchSize);
            obj.imagesFull = transform(bimdsRaw, @ColmapData.unwrapBlockCell);

            % ------------------------------------------------------------------
            % Pass B — Half-resolution preprocessing
            % Resize each tile to half of blockSize for fast early convergence.
            % Training starts with this datastore (coarser = faster iterations).
            % Same InMemory + single-level pattern as Pass A.
            % ------------------------------------------------------------------
            halfPreprocessFcn = @(block) im2single( ...
                imresize(block.Data, halfBlockSize / 2, 'bilinear'));

            bimHalfArray = blockedImage.empty(0, 1);
            for k = 1:limit_img
                bimHalfArray(k) = apply( ...
                    bimArray(k), ...
                    halfPreprocessFcn, ...
                    'BlockSize',        halfPyramidSize, ...   % [H W C]
                    'BatchSize',        miniBatchSize, ...
                    'PadPartialBlocks', true, ...
                    'PadMethod',        'symmetric', ...
                    'Adapter',          images.blocked.InMemory);
            end

            blsHalf = selectBlockLocations(bimHalfArray, ...
                'Levels',    1, ...                % single-level output from apply()
                'BlockSize', halfPyramidSize);     % [H W C]

            bimdsRawHalf = blockedImageDatastore(bimHalfArray, ...
                'BlockLocationSet', blsHalf, ...
                'PadPartialBlocks', true, ...
                'PadMethod',        'symmetric', ...
                'ReadSize',         miniBatchSize);
            obj.imagesHalf = transform(bimdsRawHalf, @ColmapData.unwrapBlockCell);

            % Start training at half resolution for fast early convergence.
            % Call obj.setLevel(1) at levelSwitchEpoch to swap to full res.
            obj.images = obj.imagesHalf;

            % ------------------------------------------------------------------
            % 6. Build Camera Label Datastores
            % arrayDatastore provides efficient random-access shuffling of
            % camera parameters combined with the image tile datastore.
            % ------------------------------------------------------------------
            obj.cameras = combine( ...
                arrayDatastore([temp_cameras.id],                         'IterationDimension', 2), ...
                arrayDatastore([temp_cameras.width],                      'IterationDimension', 2), ...
                arrayDatastore([temp_cameras.height],                     'IterationDimension', 2), ...
                arrayDatastore([temp_cameras.fx],                         'IterationDimension', 2), ...
                arrayDatastore([temp_cameras.fy],                         'IterationDimension', 2), ...
                arrayDatastore([temp_cameras.cx],                         'IterationDimension', 2), ...
                arrayDatastore([temp_cameras.cy],                         'IterationDimension', 2), ...
                arrayDatastore(reshape([temp_cameras.Rcw], 3, 3, []),     'IterationDimension', 3), ...
                arrayDatastore(reshape([temp_cameras.tcw], 3, 1, []),     'IterationDimension', 3), ...
                arrayDatastore(reshape([temp_cameras.twc], 3, 1, []),     'IterationDimension', 3));

            % ------------------------------------------------------------------
            % 7. Find Scene Scale
            % ------------------------------------------------------------------
            obj.scene_scale = arrayDatastore( ...
                ColmapData.findSceneScale(temp_cameras), 'IterationDimension', 2);

            fprintf('ColmapData initialisation complete.\n');
        end

        function setLevel(obj, level)
            % Switch the active blockedImageDatastore to a different resolution.
            %
            % Because apply() always produces single-level blockedImage output,
            % two separate datastores are pre-built at construction time.
            % setLevel swaps obj.images between them — no data is re-loaded.
            %
            % Args:
            %   level (int): 1 = full resolution (imagesFull)
            %                2 = half resolution (imagesHalf)
            %
            % Usage (in trainGaussianSplat.m):
            %   if epoch == levelSwitchEpoch
            %       obj.data.setLevel(1);
            %   end

            if level == 1
                obj.images = obj.imagesFull;
                fprintf('Switched to full-resolution blockedImageDatastore.\n');
            elseif level == 2
                obj.images = obj.imagesHalf;
                fprintf('Switched to half-resolution blockedImageDatastore.\n');
            else
                error('ColmapData:setLevel', ...
                    'Invalid level %d. Use 1 (full res) or 2 (half res).', level);
            end
        end

        function g = initGaussiansFrom3dPoints(obj, pts)
            % Initialise Gaussian parameters from the sparse 3D point cloud.
            num_pts = length(pts);

            if num_pts == 0
                g.pws = []; g.shs = []; g.rots = []; g.scales = []; g.alphas = [];
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
            % Compute scene scale as 1.1× the maximum camera-centre spread.
            if isempty(cameras)
                scale = 1.0;
                return;
            end
            twcs   = [cameras.twc];
            center = mean(twcs, 2);
            dists  = sqrt(sum((twcs - center).^2, 1));
            scale  = max(dists) * 1.1;
        end

        function R = qVec2RotMat(qvec)
            % Convert quaternion [w, x, y, z] to 3×3 rotation matrix.
            w = qvec(1); x = qvec(2); y = qvec(3); z = qvec(4);
            R = zeros(3, 3);
            R(1,1) = 1 - 2*y^2 - 2*z^2;  R(1,2) = 2*x*y - 2*w*z;      R(1,3) = 2*x*z + 2*w*y;
            R(2,1) = 2*x*y + 2*w*z;       R(2,2) = 1 - 2*x^2 - 2*z^2;  R(2,3) = 2*y*z - 2*w*x;
            R(3,1) = 2*x*z - 2*w*y;       R(3,2) = 2*y*z + 2*w*x;      R(3,3) = 1 - 2*x^2 - 2*y^2;
        end

        function imgOut = unwrapBlockCell(data)
            % blockedImageDatastore returns a cell array of H×W×C blocks.
            % This transform unwraps a single-element cell into a plain
            % H×W×C single array compatible with combine() and minibatchqueue.
            if iscell(data)
                imgOut = data{1};
            else
                imgOut = data;
            end
            % Ensure single precision for GPU transfer
            if ~isa(imgOut, 'single')
                imgOut = im2single(imgOut);
            end
        end
    end
end