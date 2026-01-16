classdef ColmapLoader
    % ColmapLoader - A utility class to load COLMAP binary export files in MATLAB.
    %
    % This class converts the functionality of colmap_loader.hpp to MATLAB.
    % It loads 'cameras.bin', 'images.bin', and 'points3D.bin' from a given directory.
    %
    % Usage:
    %   folderPath = 'path/to/dense/sparse';
    %   [cameras, images, points] = ColmapLoader.load(folderPath);
    %
    %   % Accessing data
    %   myCamera = cameras(1);  % Access camera by ID
    %   myImage = images(1);    % Access image by ID
    %   plot3([points.x], [points.y], [points.z], '.');

    methods (Static)
        function [cameras, images, points] = load(folderPath)
            % LOAD Loads cameras, images, and 3D points from a COLMAP output folder.
            %
            % Args:
            %   folderPath (char/string): Path to the folder containing .bin files
            %
            % Returns:
            %   cameras (containers.Map): Map of camera_id -> struct
            %   images  (containers.Map): Map of image_id -> struct
            %   points  (struct array): Array containing 3D point data
            
            % Paths to binary files
            camerasFile = fullfile(folderPath, 'cameras.bin');
            imagesFile = fullfile(folderPath, 'images.bin');
            pointsFile = fullfile(folderPath, 'points3D.bin');

            % Verify existence
            if ~isfile(camerasFile), error('File not found: %s', camerasFile); end
            if ~isfile(imagesFile), error('File not found: %s', imagesFile); end
            if ~isfile(pointsFile), error('File not found: %s', pointsFile); end

            % Load Data
            fprintf('Loading cameras...\n');
            cameras = ColmapLoader.loadCameras(camerasFile);
            
            fprintf('Loading images...\n');
            images = ColmapLoader.loadImages(imagesFile);
            
            fprintf('Loading 3D points...\n');
            points = ColmapLoader.loadPoints(pointsFile);
        end

        function cameras = loadCameras(filePath)
            fid = fopen(filePath, 'rb', 'l'); % Little-endian
            if fid == -1, error('Cannot open %s', filePath); end
            
            try
                numCameras = fread(fid, 1, 'uint64');
                cameras = containers.Map('KeyType', 'int32', 'ValueType', 'any');
                
                for i = 1:numCameras
                    c = struct();
                    c.id = fread(fid, 1, 'int32');
                    c.model_id = fread(fid, 1, 'int32');
                    c.w = fread(fid, 1, 'uint64');
                    c.h = fread(fid, 1, 'uint64');
                    
                    np = ColmapLoader.numParams(c.model_id);
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
                images = containers.Map('KeyType', 'int32', 'ValueType', 'any');
                
                for i = 1:numImages
                    im = struct();
                    im.id = fread(fid, 1, 'int32');
                    
                    % Quaternion (w, x, y, z)
                    im.q = fread(fid, 4, 'double');
                    
                    % Translation (tx, ty, tz)
                    im.t = fread(fid, 3, 'double');
                    
                    im.camera_id = fread(fid, 1, 'int32');
                    im.name = ColmapLoader.readString(fid);
                    
                    % Points 2D
                    numPoints2D = fread(fid, 1, 'uint64');
                    
                    % Skip 2D points data (x: double, y: double, id: int64)
                    % 8 bytes + 8 bytes + 8 bytes = 24 bytes per point
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
                    % Preallocate struct array for speed
                    % Initialize with the last element
                    points(numPoints).id = int64(0); 
                    
                    for i = 1:numPoints
                        points(i).id = fread(fid, 1, 'int64');
                        xyz = fread(fid, 3, 'double');
                        points(i).x = xyz(1);
                        points(i).y = xyz(2);
                        points(i).z = xyz(3);
                        
                        rgb = fread(fid, 3, 'uint8');
                        points(i).r = rgb(1);
                        points(i).g = rgb(2);
                        points(i).b = rgb(3);
                        
                        points(i).error = fread(fid, 1, 'double');
                        
                        trackLen = fread(fid, 1, 'uint64');
                        
                        % Skip track (image_id: int32, point2D_idx: int32)
                        % 4 bytes + 4 bytes = 8 bytes per track element
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
            % Returns number of parameters based on COLMAP model_id
            switch model_id
                case 0, n = 3; % SIMPLE_PINHOLE
                case 1, n = 4; % PINHOLE
                case 2, n = 4; % SIMPLE_RADIAL
                case 3, n = 5; % RADIAL
                case 4, n = 8; % OPENCV
                case 5, n = 12; % OPENCV_FISHEYE
                case 6, n = 12; % FULL_OPENCV
                case 7, n = 5; % FOV
                otherwise
                    error('Unknown camera model ID: %d', model_id);
            end
        end
        
        function s = readString(fid)
            % Reads a null-terminated string char-by-char
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