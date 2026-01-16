classdef ColmapData < handle
    % ColmapData - MATLAB implementation for processing COLMAP data for Gaussian Splatting
    %
    % Logic derived from gsplat_data.hpp:
    % 1. Loads Cameras, Images, and Points3D using ColmapLoader.
    % 2. Initializes Gaussians from Points3D.
    % 3. Stores images as a TransformedDatastore for lazy loading and processing.
    % 4. Computes scene scale.
    %
    % Usage:
    %   data = GsplatData('path/to/dataset', 100000, 500);
    %   
    %   % Accessing an image (lazy loaded, resized, and normalized)
    %   img1 = read(data.images); 
    %   % or
    %   img_specific = preview(data.images);

    properties
        cameras      % Struct array: {id, width, height, fx, fy, cx, cy, Rcw, tcw, twc, image_path}
        images       % TransformedDatastore
        gaussians    % Struct: {pws, shs, scales, rots, alphas}
        scene_scale  % Float: Scene scale factor
    end

    properties (Constant)
        SH_C0_0 = 0.28209479177387814;
        kScaleDownFactor = 2.0;       % Downsample factor for loading
        kInitialAlpha = 0.8;          % Initial opacity for Gaussians
    end

    methods
        function obj = ColmapData(dataset_path, max_num_gaussians, max_num_images)
            % Constructor equivalent to GuassianSplatData(...)
            %
            % Args:
            %   dataset_path (string): Root directory of the dataset
            %   max_num_gaussians (int): Maximum limit for points
            %   max_num_images (int): Maximum limit for images
            
            % 1. Load COLMAP Data
            % Assumes standard structure: /sparse/0/ and /images/
            sparse_path = fullfile(dataset_path, 'sparse', '0');
            images_dir = fullfile(dataset_path, 'images');
            
            fprintf('Loading COLMAP data from %s...\n', sparse_path);
            [cams_map, ims_map, pts_struct] = ColmapLoader.load(sparse_path);
            
            fprintf('Loaded: %d cameras, %d images, %d points3D\n', ...
                cams_map.Count, ims_map.Count, length(pts_struct));
             
            % 2. Process Metadata (Cameras and File Paths)
            % Unlike the previous version, we do not load pixel data yet.
            % We iterate to validate files and calculate camera parameters.
            
            % Get image keys and sort them to match C++ iteration stability
            im_keys = cell2mat(keys(ims_map));
            im_keys = sort(im_keys);
            
            temp_cameras = [];
            valid_image_paths = [];
            
            fprintf('Processing metadata...\n');
            for k = 1:length(im_keys)
                im = ims_map(im_keys(k));
                
                % Load Image
                full_im_path = fullfile(images_dir, im.name);
                if ~isfile(full_im_path)
                    warning('Image not found: %s', full_im_path);
                    continue;
                end
                
                % We need dimensions to safely set up the Camera intrinsics.
                % Using imfinfo is faster than imread for just dimensions.
                info = imfinfo(full_im_path);
                raw_w = info.Width;
                raw_h = info.Height;
                
                % Calculate Dimensions based on Downscale Factor
                % C++: resize by 1.0 / kScaleDownFactor
                w_cur = round(raw_w / obj.kScaleDownFactor);
                h_cur = round(raw_h / obj.kScaleDownFactor);
                
                % Retrieve Camera Model
                colmap_cam = cams_map(im.camera_id);
                w_model = double(colmap_cam.w);
                h_model = double(colmap_cam.h);
                
                w_scale = w_cur/w_model;
                h_scale = h_cur/h_model;
                
                % Create Camera Struct
                cam = struct();
                cam.id = single(im.id);
                cam.width = single(w_cur);
                cam.height = single(h_cur);
                
                % Adjust Intrinsics
                params = colmap_cam.params;
                if length(params) >= 4
                    cam.fx = single(params(1)*w_scale);
                    cam.fy = single(params(2)*h_scale);
                    cam.cx = single(params(3)*w_scale);
                    cam.cy = single(params(4)*h_scale);
                elseif length(params) == 3 
                    cam.fx = single(params(1)*w_scale);
                    cam.fy = single(params(1)*h_scale);
                    cam.cx = single(params(2)*w_scale);
                    cam.cy = single(params(3)*h_scale);
                end
                
                % Extrinsics
                cam.Rcw = single(ColmapData.qVec2RotMat(im.q)); 
                cam.tcw = single(im.t(:));                       
                cam.twc = single(-cam.Rcw'*cam.tcw);
                
                cam.image_path = full_im_path;
                
                % Store valid path and camera struct
                valid_image_paths = [valid_image_paths; full_im_path];
                if isempty(temp_cameras)
                    temp_cameras = cam;
                else
                    temp_cameras = [temp_cameras; cam];
                end
            end
            
            % 3. Initialize Gaussians from Points
            fprintf('Initializing Gaussians...\n');
            obj.gaussians = obj.initGaussiansFrom3dPoints(pts_struct);
            
            % 4. Limit and Shuffle Data (Matching C++ Logic)
            
            % Shuffle Gaussians
            num_points = size(obj.gaussians.pws, 1);
            p_idx = randperm(num_points);
            limit_g = min(num_points, max_num_gaussians);
            keep_idx = p_idx(1:limit_g);
            
            obj.gaussians.pws = obj.gaussians.pws(keep_idx, :);
            obj.gaussians.shs = obj.gaussians.shs(keep_idx, :);
            obj.gaussians.scales = obj.gaussians.scales(keep_idx, :);
            obj.gaussians.rots = obj.gaussians.rots(keep_idx, :);
            obj.gaussians.alphas = obj.gaussians.alphas(keep_idx, :);
            
            % Reverse Images/Cameras
            valid_image_paths = flip(valid_image_paths);
            temp_cameras = flip(temp_cameras);
            
            % Limit Images/Cameras
            limit_img = min(length(valid_image_paths), max_num_images);
            valid_image_paths = valid_image_paths(1:limit_img);
            temp_cameras = temp_cameras(1:limit_img);

            % 5. Create ImageDatastore
            % Create the base datastore with the filtered list of files
            % and transformed Datastore for the resizing and normalization logic
            obj.cameras = combine(arrayDatastore([temp_cameras.id], 'IterationDimension', 2),...
                arrayDatastore([temp_cameras.width], 'IterationDimension', 2),...
                arrayDatastore([temp_cameras.height], 'IterationDimension', 2),...
                arrayDatastore([temp_cameras.fx], 'IterationDimension', 2),...
                arrayDatastore([temp_cameras.fy], 'IterationDimension', 2),...
                arrayDatastore([temp_cameras.cx], 'IterationDimension', 2),...
                arrayDatastore([temp_cameras.cy], 'IterationDimension', 2),...
                arrayDatastore(reshape([temp_cameras.Rcw],3,3,[]), 'IterationDimension', 3),...
                arrayDatastore(reshape([temp_cameras.tcw],3,1,[]), 'IterationDimension', 3),...
                arrayDatastore(reshape([temp_cameras.twc],3,1,[]), 'IterationDimension', 3));
            obj.images = transform(imageDatastore(valid_image_paths), @(data) ColmapData.preprocessImage(data, obj.kScaleDownFactor));
            
            % 6. Find Scene Scale
            obj.scene_scale = arrayDatastore(ColmapData.findSceneScale(temp_cameras), 'IterationDimension', 2);
        end
        
        function g = initGaussiansFrom3dPoints(obj, pts)
            % Initialize Gaussian parameters based on sparse point cloud
            num_pts = length(pts);
            
            if num_pts == 0
                g.pws = []; g.shs = []; g.rots = []; g.scales = []; g.alphas = [];
                return;
            end
            
            % Position
            g.pws = [[pts.x]', [pts.y]', [pts.z]'];
            
            % SH Colors
            % rgb -> float[0,1] -> shift 0.5 -> div SH_C0_0
            rgb = [[pts.r]', [pts.g]', [pts.b]'];
            g.shs = ((double(rgb) / 255.0) - 0.5) / obj.SH_C0_0;
            
            % Rotation
            % C++ sets {0, 0, 0, 1}
            g.rots = repmat([0, 0, 0, 1], num_pts, 1);
            
            % Alpha
            g.alphas = repmat(obj.kInitialAlpha, num_pts, 1);
            
            % Scales
            % C++ uses nearest_dist placeholder = 0.1f
            nearest_dist = 0.1;
            g.scales = repmat([nearest_dist, nearest_dist, nearest_dist], num_pts, 1);
        end
    end
    
    methods (Static)
        function imgOut = preprocessImage(imgIn, scaleDownFactor)
            % The transformation function applied to every image read from the datastore.
            % 1. Resize
            % 2. Normalize to [0, 1]
            
            % If reading a batch, imgIn might be a cell or array, but with transforming
            % an ImageDatastore individually, it usually passes one image at a time unless ReadSize is changed.
            
            if iscell(imgIn)
                imgIn = imgIn{1};
            end
            
            % Resize
            imgResized = imresize(imgIn, 1.0 / scaleDownFactor, 'bilinear');
            
            % Convert to single [0, 1]
            imgOut = im2single(imgResized);
        end
        
        function scale = findSceneScale(cameras)
            if isempty(cameras)
                scale = 1.0;
                return;
            end
            
            % Extract Camera Centers (twc)
            % cameras is struct array, field twc is 3x1
            twcs = [cameras.twc]; % 3 x N matrix
            center = mean(twcs, 2);
            dist_vecs = twcs - center;
            dists = sqrt(sum(dist_vecs.^2, 1));
            
            kScaleFactor = 1.1;
            scale = max(dists)*kScaleFactor;
            
            fprintf('scene scale: %f\n', scale);
        end

        function R = qVec2RotMat(qvec)
            % Standard Quaternion to Matrix conversion [w, x, y, z]
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