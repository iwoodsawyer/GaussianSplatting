classdef ColmapData < handle
    % ColmapData - MATLAB implementation for processing COLMAP data for
    % 3D Gaussian Splatting.
    %
    % Logic derived from gsplat_data.hpp:
    %   1. Loads Cameras, Images, and Points3D using ColmapLoader.
    %   2. Initializes Gaussians from Points3D.
    %   3. Stores images as a blockedImageDatastore: each read() returns one
    %      fixed-size block (blockSize + 2*overlap), not a full image.
    %   4. Two resolution levels are pre-built:
    %        imagesHalf/camerasHalf : downscaled by kScaleDownFactor*2
    %        imagesFull/camerasFull : downscaled by kScaleDownFactor
    %      obj.images/obj.cameras start pointing to the Half level.
    %      Call obj.setLevel(1) to switch to the Full level mid-training.
    %   5. Computes scene scale.
    %
    % Design note — camera info repeated per block:
    %   blockedImageDatastore returns N blocks per image, while COLMAP has
    %   exactly 1 camera entry per image. To preserve 1-to-1 correspondence
    %   in combine(images, cameras), obj.cameras repeats each image's camera
    %   entry once per block, with cx/cy shifted to that block's local
    %   origin. The per-block pixel offset is computed from the datastore's
    %   own BlockLocationSet via sub2world/world2sub, so it always matches
    %   the datastore's actual read order. See buildBlockedLevel().
    %
    % Usage:
    %   data = ColmapData('path/to/dataset', 2000, 20, [128 128], [0 0]);
    %   data.setLevel(1);   % switch to full resolution mid-training
    %   img = read(data.images);

    properties
        cameras      % Combined arrayDatastore: id,blockId,blockRow,blockCol,w,h,fx,fy,cx,cy,Rcw,tcw,twc
        images       % blockedImageDatastore currently active (half or full res)
        gaussians    % Struct: {pws, shs, scales, rots, alphas}
        scene_scale  % arrayDatastore: scene scale factor (float)
        blockSize    % [H W] block size fed to blockedImage/blockedImageDatastore
        overlap      % [H W] BorderSize (halo) added around each block
    end

    properties (Access = private)
        % Two independent blockedImageDatastore/camera pairs for resolution
        % scheduling. Each pair is built once in the constructor since block
        % layout (and therefore per-block camera offsets) differs per level.
        imagesHalf   % blockedImageDatastore at half resolution
        imagesFull   % blockedImageDatastore at full resolution
        camerasHalf  % Combined arrayDatastore matching imagesHalf
        camerasFull  % Combined arrayDatastore matching imagesFull
    end

    properties (Constant)
        SH_C0_0          = 0.28209479177387814; % Zeroth-order SH coefficient
        kScaleDownFactor = 2.0;                 % Base downsample factor for loading
        kInitialAlpha    = 0.8;                 % Initial Gaussian opacity
    end

    methods
        function obj = ColmapData(dataset_path, max_num_gaussians, max_num_images, blockSize, overlap)
            % Constructor: loads COLMAP data, initialises Gaussians, and
            % builds two blockedImageDatastore levels for the resolution schedule.
            %
            % Args:
            %   dataset_path      (string): Root directory of the dataset.
            %   max_num_gaussians (int)   : Maximum number of Gaussians to keep.
            %   max_num_images    (int)   : Maximum number of training images.
            %   blockSize         (1x2 int): Block [H W] for blockedImage. Default [128 128].
            %   overlap           (1x2 int): Border [H W] added around each block. Default [0 0].

            if nargin < 4 || isempty(blockSize)
                blockSize = [128, 128];
            end
            if nargin < 5 || isempty(overlap)
                overlap = [0, 0];
            end
            obj.blockSize = blockSize;
            obj.overlap   = overlap;

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
            % imfinfo is faster than imread for just obtaining dimensions.
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

                % Use imfinfo for dimensions only — faster than imread
                info  = imfinfo(full_im_path);
                raw_w = info.Width;
                raw_h = info.Height;

                % Target dimensions after the base kScaleDownFactor downscale
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

                % Adjust intrinsics to the full downscaled (whole-image)
                % resolution. Per-block cx/cy shifting happens later in
                % buildBlockedLevel(), not here.
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

            % Reverse and limit images/cameras (matches original C++ ordering)
            valid_image_paths = flip(valid_image_paths);
            temp_cameras      = flip(temp_cameras);
            limit_img         = min(length(valid_image_paths), max_num_images);
            valid_image_paths = valid_image_paths(1:limit_img);
            temp_cameras      = temp_cameras(1:limit_img);

            % ------------------------------------------------------------------
            % 5. Build Blocked-Image Datastores + Per-Block Cameras
            %
            % Each read() returns one fixed-size block (blockSize + 2*overlap),
            % not a full image. Camera info is repeated once per block, with
            % cx/cy shifted to the block's local origin, so combine(images,
            % cameras) still gives strict 1-to-1 correspondence.
            %
            % imagesHalf: kScaleDownFactor*2 — faster early-epoch convergence
            % imagesFull: kScaleDownFactor   — fine-detail refinement
            % ------------------------------------------------------------------

            fprintf('Building blocked-image datastores (Half + Full resolution)...\n');

            [obj.imagesHalf, obj.camerasHalf] = obj.buildBlockedLevel( ...
                valid_image_paths, temp_cameras, obj.kScaleDownFactor * 2.0);
            [obj.imagesFull, obj.camerasFull] = obj.buildBlockedLevel( ...
                valid_image_paths, temp_cameras, obj.kScaleDownFactor);

            % Start at half resolution
            obj.images  = obj.imagesHalf;
            obj.cameras = obj.camerasHalf;

            % ------------------------------------------------------------------
            % 6. Find Scene Scale
            % ------------------------------------------------------------------
            obj.scene_scale = arrayDatastore( ...
                ColmapData.findSceneScale(temp_cameras), 'IterationDimension', 2);

            fprintf('ColmapData initialisation complete.\n');
        end

        function setLevel(obj, level)
            % Switch the active image+camera datastores to a different resolution.
            %
            % Block layout differs per resolution level, so the per-block
            % camera offsets differ too — images and cameras must be swapped
            % together. After calling setLevel, the caller must also call
            % obj.updateResolution() in GaussianSplatter to reallocate GPU
            % buffers to the new block canvas size.
            %
            % Args:
            %   level (int): 1 = full resolution  (imagesFull/camerasFull)
            %                2 = half resolution  (imagesHalf/camerasHalf)
            %
            % Usage (in trainGaussianSplat.m):
            %   if epoch == levelSwitchEpoch
            %       obj.data.setLevel(1);
            %       obj.updateResolution();
            %   end

            if level == 1
                obj.images  = obj.imagesFull;
                obj.cameras = obj.camerasFull;
                fprintf('Switched to full-resolution datastore.\n');
            elseif level == 2
                obj.images  = obj.imagesHalf;
                obj.cameras = obj.camerasHalf;
                fprintf('Switched to half-resolution datastore.\n');
            else
                error('ColmapData:setLevel', ...
                    'Invalid level %d. Use 1 (full res) or 2 (half res).', level);
            end
        end

        function [bimds, camDS] = buildBlockedLevel(obj, imagePaths, cams, scaleDownFactor)
            % Build an in-memory blockedImageDatastore for one resolution
            % level, plus a matching combined arrayDatastore that repeats
            % each image's camera entry once per block (cx/cy shifted to
            % the block's local pixel origin).
            %
            % Args:
            %   imagePaths      (Nx1 string): Paths, 1-to-1 with cams.
            %   cams            (Nx1 struct): Per-image camera metadata.
            %   scaleDownFactor (double)    : Reciprocal resize factor for
            %                                 this resolution level.

            numImgs = numel(imagePaths);
            bims = [];
            for k = 1:numImgs
                img = ColmapData.preprocessImage(imread(imagePaths(k)), scaleDownFactor);
                b   = blockedImage(img, 'BlockSize', obj.blockSize);
                if isempty(bims)
                    bims = b;
                else
                    bims(end+1) = b; %#ok<AGROW>
                end
            end

            % Full read window = core blockSize + halo on both sides. BlockOffsets
            % equal to the core blockSize (smaller than the full window) makes
            % adjacent windows overlap by 2*overlap pixels — the halo is baked
            % directly into the block grid instead of added via BorderSize.
            % ExcludeIncompleteBlocks is left false so edge windows that overflow
            % the image (common once the halo is added) are kept and zero-padded
            % by PadMethod below, instead of being dropped outright.
            blockH = obj.blockSize(1) + 2 * obj.overlap(1);
            blockW = obj.blockSize(2) + 2 * obj.overlap(2);

            bls = selectBlockLocations(bims, ...
                BlockSize=[blockH, blockW], ...
                BlockOffsets=obj.blockSize);

            bimds = blockedImageDatastore(bims, BlockLocationSet=bls, PadMethod='replicate');

            numBlocks = size(bls.BlockOrigin, 1);

            blockCams = repmat(struct('id', single(0), 'blockId', single(0), ...
                'blockRow', single(0), 'blockCol', single(0), ...
                'width', single(0), 'height', single(0), 'fx', single(0), ...
                'fy', single(0), 'cx', single(0), 'cy', single(0), ...
                'Rcw', single(zeros(3,3)), 'tcw', single(zeros(3,1)), ...
                'twc', single(zeros(3,1))), numBlocks, 1);

            for k = 1:numBlocks
                imgIdx = bls.ImageNumber(k);
                cam    = cams(imgIdx);

                % BlockOrigin columns are (x,y[,channel]); world2sub expects
                % (row,col[,channel]) order, so swap only the first two
                % columns — fliplr of an n×3 row would scramble row/col/channel.
                worldOrd = bls.BlockOrigin(k, :);
                worldOrd(1:2) = worldOrd([2 1]);
                originRC  = world2sub(bims(imgIdx), worldOrd);
                % BlockOrigin is now the full window's own top-left corner (no
                % separate core grid to offset from), so no overlap subtraction.
                rowOffset = originRC(1) - 1;
                colOffset = originRC(2) - 1;

                % 1-based tile position of this block within its source
                % image's grid — used to stitch blocks back together for
                % preview/visualization (grid spacing == blockSize).
                blockCams(k).blockRow = single(round((originRC(1) - 1) / obj.blockSize(1)) + 1);
                blockCams(k).blockCol = single(round((originRC(2) - 1) / obj.blockSize(2)) + 1);

                blockCams(k).id      = cam.id;
                blockCams(k).blockId = single(k);
                blockCams(k).width   = single(blockW);
                blockCams(k).height  = single(blockH);
                blockCams(k).fx      = cam.fx;
                blockCams(k).fy      = cam.fy;
                blockCams(k).cx      = cam.cx - single(colOffset);
                blockCams(k).cy      = cam.cy - single(rowOffset);
                blockCams(k).Rcw     = cam.Rcw;
                blockCams(k).tcw     = cam.tcw;
                blockCams(k).twc     = cam.twc;
            end

            camDS = combine( ...
                arrayDatastore([blockCams.id],                          'IterationDimension', 2), ...
                arrayDatastore([blockCams.blockId],                     'IterationDimension', 2), ...
                arrayDatastore([blockCams.blockRow],                    'IterationDimension', 2), ...
                arrayDatastore([blockCams.blockCol],                    'IterationDimension', 2), ...
                arrayDatastore([blockCams.width],                       'IterationDimension', 2), ...
                arrayDatastore([blockCams.height],                      'IterationDimension', 2), ...
                arrayDatastore([blockCams.fx],                          'IterationDimension', 2), ...
                arrayDatastore([blockCams.fy],                          'IterationDimension', 2), ...
                arrayDatastore([blockCams.cx],                          'IterationDimension', 2), ...
                arrayDatastore([blockCams.cy],                          'IterationDimension', 2), ...
                arrayDatastore(reshape([blockCams.Rcw], 3, 3, []),      'IterationDimension', 3), ...
                arrayDatastore(reshape([blockCams.tcw], 3, 1, []),      'IterationDimension', 3), ...
                arrayDatastore(reshape([blockCams.twc], 3, 1, []),      'IterationDimension', 3));
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

            % Initial rotation: identity quaternion [w x y z] = [1 0 0 0]
            % Convention: column 1 = w, columns 2-4 = x,y,z, matching qVec2RotMat
            % and projectGaussiansWithCulling in GaussianSplatter.m.
            % [0,0,0,1] is a 180° rotation around Z (R11=R22=-1) — NOT identity.
            g.rots = repmat([1, 0, 0, 0], num_pts, 1);

            % Initial opacity
            g.alphas = repmat(obj.kInitialAlpha, num_pts, 1);

            % Initial isotropic scale based on nearest-neighbour heuristic
            nearest_dist = 0.1;
            g.scales = repmat([nearest_dist, nearest_dist, nearest_dist], num_pts, 1);
        end
    end

    methods (Static)
        function imgOut = preprocessImage(imgIn, scaleDownFactor)
            % Full-image preprocessing: resize then normalise to [0, 1].
            % Applied via transform() so images are loaded lazily on demand.
            %
            % Args:
            %   imgIn         : Single image (H×W×C uint8) or cell wrapping one.
            %   scaleDownFactor: Reciprocal of the resize scale factor.
            %                   kScaleDownFactor   → full training resolution
            %                   kScaleDownFactor*2 → half training resolution

            if iscell(imgIn)
                imgIn = imgIn{1};
            end
            % imresize with 1/scaleDownFactor gives the target resolution.
            % 'bilinear' matches the apply() method used in the blocked path.
            imgResized = imresize(imgIn, 1.0 / scaleDownFactor, 'bilinear');

            % im2single: uint8 → single and divides by 255 (normalises to [0,1])
            imgOut = im2single(imgResized);
        end

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
    end
end