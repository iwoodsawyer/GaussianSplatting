classdef ColmapLoader
    % ColmapLoader - Utility class to load COLMAP binary export files in MATLAB.
    %
    % Converts the functionality of colmap_loader.hpp to MATLAB.
    % Loads 'cameras.bin', 'images.bin', and 'points3D.bin' from a given
    % directory into MATLAB data structures used by ColmapData.
    %
    % Usage:
    %   folderPath = 'path/to/sparse/0';
    %   [cameras, images, points] = ColmapLoader.load(folderPath);
    %
    %   % Access camera by ID
    %   myCamera = cameras(1);
    %   % Access image metadata by ID
    %   myImage  = images(1);
    %   % Visualise sparse point cloud
    %   plot3([points.x], [points.y], [points.z], '.');

    methods (Static)
        function [cameras, images, points] = load(folderPath)
            % LOAD  Load cameras, images, and 3D points from a COLMAP output folder.
            %
            % Args:
            %   folderPath (char/string): Path to folder containing .bin files.
            %
            % Returns:
            %   cameras (containers.Map): camera_id → struct
            %   images  (containers.Map): image_id  → struct
            %   points  (struct array)  : array of 3D point data

            camerasFile = fullfile(folderPath, 'cameras.bin');
            imagesFile  = fullfile(folderPath, 'images.bin');
            pointsFile  = fullfile(folderPath, 'points3D.bin');

            if ~isfile(camerasFile), error('File not found: %s', camerasFile); end
            if ~isfile(imagesFile),  error('File not found: %s', imagesFile);  end
            if ~isfile(pointsFile),  error('File not found: %s', pointsFile);  end

            fprintf('Loading cameras...\n');
            cameras = ColmapLoader.loadCameras(camerasFile);

            fprintf('Loading images...\n');
            images = ColmapLoader.loadImages(imagesFile);

            fprintf('Loading 3D points...\n');
            points = ColmapLoader.loadPoints(pointsFile);
        end

        function cameras = loadCameras(filePath)
            fid = fopen(filePath, 'rb', 'l');  % Little-endian
            if fid == -1, error('Cannot open %s', filePath); end

            try
                numCameras = fread(fid, 1, 'uint64');
                cameras    = containers.Map('KeyType', 'int32', 'ValueType', 'any');

                for i = 1:numCameras
                    c.id       = fread(fid, 1, 'int32');
                    c.model_id = fread(fid, 1, 'int32');
                    c.w        = fread(fid, 1, 'uint64');
                    c.h        = fread(fid, 1, 'uint64');

                    np       = ColmapLoader.numParams(c.model_id);
                    c.params = fread(fid, np, 'double');

                    cameras(c.id) = c;
                end
            catch ME
                fclose(fid);
                rethrow(ME);
            end
            fclose(fid);
        end

        function images = loadImages(filePath)
            fid = fopen(filePath, 'rb', 'l');
            if fid == -1, error('Cannot open %s', filePath); end

            try
                numImages = fread(fid, 1, 'uint64');
                images    = containers.Map('KeyType', 'int32', 'ValueType', 'any');

                for i = 1:numImages
                    im.id        = fread(fid, 1, 'int32');
                    im.q         = fread(fid, 4, 'double'); % Quaternion (w, x, y, z)
                    im.t         = fread(fid, 3, 'double'); % Translation (tx, ty, tz)
                    im.camera_id = fread(fid, 1, 'int32');
                    im.name      = ColmapLoader.readString(fid);

                    % Skip 2D keypoint observations (x: double, y: double, id: int64)
                    % Each observation: 8 + 8 + 8 = 24 bytes
                    numPoints2D = fread(fid, 1, 'uint64');
                    if numPoints2D > 0
                        fseek(fid, 24 * numPoints2D, 'cof');
                    end

                    images(im.id) = im;
                end
            catch ME
                fclose(fid);
                rethrow(ME);
            end
            fclose(fid);
        end

        function points = loadPoints(filePath)
            fid = fopen(filePath, 'rb', 'l');
            if fid == -1, error('Cannot open %s', filePath); end

            try
                numPoints = fread(fid, 1, 'uint64');

                if numPoints > 0
                    % Preallocate by initialising the last element first
                    points(numPoints).id = int64(0);

                    for i = 1:numPoints
                        points(i).id = fread(fid, 1, 'int64');

                        xyz            = fread(fid, 3, 'double');
                        points(i).x    = xyz(1);
                        points(i).y    = xyz(2);
                        points(i).z    = xyz(3);

                        rgb            = fread(fid, 3, 'uint8');
                        points(i).r    = rgb(1);
                        points(i).g    = rgb(2);
                        points(i).b    = rgb(3);

                        points(i).error = fread(fid, 1, 'double');

                        % Skip track (image_id: int32, point2D_idx: int32)
                        % Each track element: 4 + 4 = 8 bytes
                        trackLen = fread(fid, 1, 'uint64');
                        if trackLen > 0
                            fseek(fid, 8 * trackLen, 'cof');
                        end
                    end
                else
                    points = struct([]);
                end
            catch ME
                fclose(fid);
                rethrow(ME);
            end
            fclose(fid);
        end
    end

    methods (Static, Access = private)
        function n = numParams(model_id)
            % Return number of intrinsic parameters for a given COLMAP model ID.
            switch model_id
                case 0,  n = 3;  % SIMPLE_PINHOLE
                case 1,  n = 4;  % PINHOLE
                case 2,  n = 4;  % SIMPLE_RADIAL
                case 3,  n = 5;  % RADIAL
                case 4,  n = 8;  % OPENCV
                case 5,  n = 12; % OPENCV_FISHEYE
                case 6,  n = 12; % FULL_OPENCV
                case 7,  n = 5;  % FOV
                otherwise
                    error('Unknown camera model ID: %d', model_id);
            end
        end

        function s = readString(fid)
            % Read a null-terminated ASCII string from binary file handle fid.
            chars = char.empty;
            while true
                c = fread(fid, 1, 'char');
                if isempty(c) || c == 0
                    break;
                end
                chars(end+1) = char(c); %#ok<AGROW>
            end
            s = string(chars);
        end
    end
end