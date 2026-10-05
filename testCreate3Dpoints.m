function tests = testCreate3Dpoints
tests = functiontests(localfunctions);
end

function testCentresPreservePositionsAndDcColors(testCase)
    params = fixtureParams();
    result = runCloud(params, "centres", "dc", [0 0 0]);
    expected = single(0.5) + single(0.28209479177387814) .* ...
        reshape(extractdata(params.shs(:,1,:)), 3, 3);
    verifyEqual(testCase, result.ptCloud.Count, 3);
    verifyEqual(testCase, result.ptCloud.Location, extractdata(params.pws));
    verifyEqual(testCase, result.colors, expected, 'AbsTol', single(1e-6));
end

function testShColorsUseConfiguredCamera(testCase)
    params = fixtureParams();
    cameraPosition = [-1 0 0];
    result = runCloud(params, "centres", "sh", cameraPosition);
    direction = extractdata(params.pws) - single(cameraPosition);
    direction = direction ./ vecnorm(direction, 2, 2);
    sh = extractdata(params.shs);
    expected = single(0.5) + single(0.28209479177387814) .* ...
        reshape(sh(:,1,:), 3, 3) - single(0.4886025119029199) .* ...
        direction(:,1) .* reshape(sh(:,2,:), 3, 3);
    verifyEqual(testCase, result.colors, expected, 'AbsTol', single(1e-6));
end

function testSampledCovarianceMatchesRendererLayout(testCase)
    result = runCloud(fixtureParams(), "sampled", "dc", [0 0 0]);
    verifyEqual(testCase, size(result.pts, 1), numel(result.gidx));
    verifyGreaterThan(testCase, result.ptCloud.Count, 0);
    % The last (partial) batch contains the third Gaussian.
    q = double(result.quat(3,:));
    qw = q(1); qx = q(2); qy = q(3); qz = q(4);
    row = [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy), ...
           2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx), ...
           2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)];
    R = reshape(row', 3, 3);
    M = R * diag(double(result.scale(3,:)));
    expected = M * M';
    actual = cov(double(result.pts));
    verifyLessThan(testCase, norm(actual - expected, 'fro') / norm(expected, 'fro'), 0.05);
end

function testLargeCountsStayExact(testCase)
    root = fileparts(fileparts(mfilename('fullpath')));
    src = fileread(fullfile(root, 'create3Dpoints.m'));
    statement = regexp(src, '(?m)^\s*nPtsPer = [^\n]+', 'match', 'once');
    scale = single([2^24 1 1; 1 1 1]);
    alpha = ones(2, 1, 'single');
    pointsPerSplat = 1;
    eval(statement);
    verifyClass(testCase, nPtsPer, 'double');
    verifyEqual(testCase, sum(nPtsPer), 2^24 + 1);
end

function testEmptyCloudReportsError(testCase)
    params = fixtureParams();
    params.alphas_raw = dlarray(single([-100; -100; -100]));
    verifyError(testCase, @() runCloud(params, "centres", "dc", [0 0 0]), ...
        'create3Dpoints:EmptyCloud');
end

function testCameraPositionUsesColmapImageId(testCase)
    datasetPath = tempname;
    imagesDir = fullfile(datasetPath, 'sparse', '0');
    mkdir(imagesDir);
    cleanupDataset = onCleanup(@() rmdir(datasetPath, 's'));

    fid = fopen(fullfile(imagesDir, 'images.bin'), 'w', 'ieee-le');
    fwrite(fid, uint64(1), 'uint64');
    fwrite(fid, int32(7), 'int32');
    fwrite(fid, [1 0 0 0], 'double');
    fwrite(fid, [-1 2 -3], 'double');
    fwrite(fid, int32(1), 'int32');
    fwrite(fid, uint8([double('test.jpg') 0]), 'uint8');
    fwrite(fid, uint64(0), 'uint64');
    fclose(fid);

    result = runCloud(fixtureParams(), "centres", "sh", [0 0 0], 7, datasetPath);
    verifyEqual(testCase, result.cameraPosition, [1 -2 3]);
end

function params = fixtureParams()
    params.pws = dlarray(single([0 0 0; 1 0 0; 2 0 0]));
    sh = zeros(3, 9, 3, 'single');
    sh(:,1,:) = repmat(reshape(single([0.1 0.2 0.3]), 1, 1, 3), 3, 1, 1);
    sh(:,2,:) = 0.2;
    params.shs = dlarray(sh);
    params.scales_raw = dlarray(log(repmat(single([0.15 0.02 0.01]), 3, 1)));
    params.rots_raw = dlarray(repmat(single([cos(pi/8) 0 0 sin(pi/8)]), 3, 1));
    params.alphas_raw = dlarray(repmat(single(log(4)), 3, 1));
end

function result = runCloud(params, mode, color, camera, imageId, datasetPath)
    filename = [tempname '.mat'];
    save(filename, 'params');
    cleanupFile = onCleanup(@() delete(filename));
    previousRng = rng;
    cleanupRng = onCleanup(@() rng(previousRng));
    rng(0);
    root = fileparts(fileparts(mfilename('fullpath')));
    previousPath = path;
    addpath(root);
    cleanupPath = onCleanup(@() path(previousPath));
    src = fileread(fullfile(root, 'create3Dpoints.m'));
    src = strrep(src, 'clear; clc;', '');
    src = strrep(src, 'filename       = ''gaussians.mat'';', ...
        ['filename = ''' strrep(filename, '''', '''''') ''';']);
    src = strrep(src, 'cloudMode      = "centres";', ...
        ['cloudMode = "' char(mode) '";']);
    src = strrep(src, 'colorMode      = "dc";', ...
        ['colorMode = "' char(color) '";']);
    src = strrep(src, 'cameraPosition = [0 0 0];', ...
        ['cameraPosition = ' mat2str(camera) ';']);
    if nargin < 5
        imageId = [];
        datasetPath = '';
    end
    src = regexprep(src, '(?m)^imageId\s*=[^\n]*$', ...
        regexptranslate('escape', ['imageId = ' mat2str(imageId) ';']));
    src = regexprep(src, '(?m)^datasetPath\s*=[^\n]*$', ...
        regexptranslate('escape', ['datasetPath = ''' strrep(datasetPath, '''', '''''') ''';']));
    src = strrep(src, 'pointsPerSplat = 1e5;', 'pointsPerSplat = 1e9;');
    src = strrep(src, 'batchSize      = 5000;', 'batchSize = 2;');
    stop = strfind(src, '% Display with controllable axis limits');
    eval(src(1:stop(1)-1));
    result = struct('ptCloud', ptCloud, 'colors', colors, 'quat', quat, ...
        'scale', scale, 'cameraPosition', cameraPosition);
    if strcmp(mode, "sampled")
        result.pts = pts;
        result.gidx = gidx;
    end
end
