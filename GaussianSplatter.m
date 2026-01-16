classdef GaussianSplatter < handle
    % GaussianSplattingTrainer 
    % MATLAB implementation of 3D Gaussian Splatting training loop.

    properties (Constant, Access=private)
        % Window Size
        windowSize = single(11);

        % Constants
        C1 = single(0.01^2);
        C2 = single(0.03^2);

        % Loss Weights
        lambda = single(0.2);
    end

    properties
        % Sizes
        numImages
        numGaussians
        miniBatchSize
        imageWidth
        imageHeight
        imageChannel

        % Data
        datasetPath
        data

        % Learnable Parameters
        params

        % Densification
        sceneScale
        scaleThresh = single(0.1); % Relative to scene size
        forcePruneThreshold = single(-4.0); % sigmoid(-4) approx 0.018
        forceKeepThreshold = single(2.0); % sigmoid(1) approx 0.7
        splitScaleFactor = single(0.2); % Reduce scale when split with log(1.6) approx 0.2

        % Storage
        image
        image_gt
        camera
        gaussians
        window
        X
        Y
        XY
        T

        % Kernel Functions
        fcnValid
    end
    
    methods
        function this = GaussianSplatter(datasetPath, numGaussians, numImages)
            % Constructor: Load data and initialize params
            this.camera = struct;

            % Store sizes
            this.datasetPath = datasetPath;
            this.numGaussians = numGaussians;
            this.numImages = numImages;

            % Load the data as an image datastore and camera labels as array datastore.
            this.data = ColmapData(datasetPath, numGaussians, numImages);

            % Get images sizes
            [this.imageHeight,this.imageWidth,this.imageChannel] = size(preview(this.data.images));

            % Create Learnable Parameters
            this.params = this.createLearnableParams(this.data.gaussians);

            % Create window (Gaussian Kernel) for SSIM loss function
            this.window = this.createWindow(this.windowSize, this.imageChannel);

            % Calculate scale threshold for densification
            temp = preview(this.data.scene_scale);
            this.sceneScale = temp{1};
        end

        function initStorage(this,miniBatchSize)
            % Initialize storage

            % Store sizes
            this.miniBatchSize = miniBatchSize;

            % Initialize Storage
            this.image = dlarray(zeros(this.imageHeight,this.imageWidth,3,this.miniBatchSize,'single'),'SSCB');
            this.window = repmat(this.window,1,1,1,miniBatchSize);
            [this.X, this.Y] = meshgrid(1:uint32(this.imageWidth), 1:uint32(this.imageHeight));
            this.XY = dlarray(repmat(single(this.X.^2 + this.Y.^2),1,1,1,this.miniBatchSize), 'SSCB');
            this.X = dlarray(repmat(single(this.X),1,1,1,this.miniBatchSize), 'SSCB');
            this.Y = dlarray(repmat(single(this.Y),1,1,1,this.miniBatchSize), 'SSCB');
            this.T = dlarray(ones(this.imageHeight,this.imageWidth,1,this.miniBatchSize,'single'), 'SSCB');

            % Kernel Function
            this.fcnValid = @(x,y,z) (z > single(0.2)) && (z < single(100.0)) &&...
                (x > -single(this.imageWidth)*single(0.5)) && (x < single(this.imageWidth)*single(1.5)) && ...
                (y > -single(this.imageHeight)*single(0.5)) && (y < single(this.imageHeight)*single(1.5));

            % Move to GPU if input is on GPU
            if isgpuarray(this.camera.id)
                this.params.pws = gpuArray(this.params.pws);
                this.params.shs = gpuArray(this.params.shs);
                this.params.scales_raw = gpuArray(this.params.scales_raw);
                this.params.alphas_raw = gpuArray(this.params.alphas_raw);
                this.params.rots_raw = gpuArray(this.params.rots_raw);

                this.image = gpuArray(this.image); 
                this.window = gpuArray(this.window);
                this.X = gpuArray(this.X); 
                this.Y = gpuArray(this.Y); 
                this.T = gpuArray(this.T); 
            end
        end

        function [loss,grads] = modelStep(this,params)
            % Forward pass and loss calculation

            % Forward Pass
            this.camera = structfun(@stripdims,this.camera,'UniformOutput',false);
            this.projectGaussiansWithCulling(params);

            % L1 Loss
            l1_loss = mean(abs(this.image - this.image_gt), 'all');

            % SSIM Loss (Differentiable)
            ssim_loss = 1.0 - this.ssimLoss();

            % Combined Loss
            loss = (1.0 - this.lambda)*l1_loss + this.lambda*ssim_loss;

            % Compute Gradients
            grads = dlgradient(loss, params);
        end

        function projectGaussiansWithCulling(this,params)
            % Projects Gaussians with Frustum Culling and renders them

            % Project to Camera Space
            this.gaussians = pagemtimes(repmat(params.pws,1,1,this.miniBatchSize),'none',this.camera.Rcw,'transpose');
            this.gaussians = this.gaussians + repmat(pagetranspose(reshape(this.camera.tcw,3,1,this.miniBatchSize)),size(params.pws,1),1,1);

            % Project to Image Plane (Homogeneous division)
            %inv_z = arrayfun(this.fcnInv, this.gaussians(:,3,:));
            inv_z = single(1.0)./max(this.gaussians(:,3,:),single(1e-6));
            this.gaussians(:,1,:) = this.gaussians(:,1,:).*inv_z.*reshape(this.camera.fx,1,1,this.miniBatchSize) + reshape(this.camera.cx,1,1,this.miniBatchSize);
            this.gaussians(:,2,:) = this.gaussians(:,2,:).*inv_z.*reshape(this.camera.fy,1,1,this.miniBatchSize) + reshape(this.camera.cy,1,1,this.miniBatchSize);

            % Frustum Culling
            %valid = arrayfun(this.fcnValid,this.gaussians(:,1,:),this.gaussians(:,2,:),this.gaussians(:,3,:));
            valid = (this.gaussians(:,3,:) > single(0.2)) &...
                (this.gaussians(:,3,:) < single(100.0)) &...
                (this.gaussians(:,1,:) > -single(this.imageWidth)*single(0.5)) &...
                (this.gaussians(:,1,:) < single(this.imageWidth)*single(1.5)) & ...
                (this.gaussians(:,2,:) > -single(this.imageHeight)*single(0.5)) &...
                (this.gaussians(:,2,:) < single(this.imageHeight)*single(1.5));
   
            % Reset image
            this.image(:) = 0;
            this.T(:) = 1;

            % If no guassians splat in 2d image return empty image
            nr_valid = sum(valid,1);
            if ~any(nr_valid)
                return;
            end

            % Extract valid subsets
            this.gaussians(:,1,:) = this.gaussians(:,1,:).*valid;
            this.gaussians(:,2,:) = this.gaussians(:,2,:).*valid;
            this.gaussians(:,3,:) = this.gaussians(:,3,:).*valid;
            alphas = dlarray(reshape(repmat(single(1.0)./(single(1.0) + exp(-params.alphas_raw)),1,1,this.miniBatchSize).*valid,this.numGaussians,1,1,this.miniBatchSize),'SSCB');

            % Compute Covariance 3D -> 2D using simplified 2D radii (Projected Splat)
            radius = dlarray(reshape(repmat(max(exp(params.scales_raw),[],2),1,1,this.miniBatchSize).*inv_z.*reshape(this.camera.fx,1,1,this.miniBatchSize),this.numGaussians,1,1,this.miniBatchSize),'SSCB');

            % Render (Splatting) using "Weighted Sum of Gaussians at Pixel Centers"
            colors = dlarray(reshape(repmat(max(min(single(0.5) + single(0.28209479177387814).*params.shs,single(1.0)),single(0.0)),1,1,this.miniBatchSize),this.numGaussians,1,3,this.miniBatchSize),'SSCB');

            % Sort by depth (Painter's Algorithm)
            u = dlarray(reshape(this.gaussians(:,1,:),this.numGaussians,1,1,this.miniBatchSize), 'SSCB');
            v = dlarray(reshape(this.gaussians(:,2,:),this.numGaussians,1,1,this.miniBatchSize), 'SSCB');
            [~, sortIdx] = sort(this.gaussians(:,3,:), 'descend');
            for k = 1:this.miniBatchSize
                u(:,:,:,k) = u(sortIdx(:,:,k),:,:,k);
                v(:,:,:,k) = v(sortIdx(:,:,k),:,:,k);
                radius(:,:,:,k) = radius(sortIdx(:,:,k),:,:,k);
                colors(:,:,1,k) = colors(sortIdx(:,:,k),:,1,k);
                colors(:,:,2,k) = colors(sortIdx(:,:,k),:,2,k);
                colors(:,:,3,k) = colors(sortIdx(:,:,k),:,3,k);
                alphas(:,:,:,k) = alphas(sortIdx(:,:,k),:,:,k);
            end

            % A simplified Gaussian Splatting Rasterizer compatible with dlarray.
            % This is brute-force compared to Tile-Based but functional for logic conversion.

            % Processing Loop
            for k = 1:max(nr_valid)
                % Gaussian falloff to mask to ignore far pixels
                alphaT = this.XY;
                alphaT = alphaT - (single(2.0).*u(k,:,:,:)).*this.X;
                alphaT = alphaT - (single(2.0).*v(k,:,:,:)).*this.Y;
                alphaT = alphaT + u(k,:,:,:).^single(2.0);
                alphaT = alphaT + v(k,:,:,:).^single(2.0);
                alphaT = alphaT.*(-single(1.5)./max(radius(k,:,:,:).^2,single(1e-6)));
                alphaT = exp(alphaT);
                alphaT = alphas(k,:,:,:).*alphaT; 
                alphaT = this.T.*alphaT;

                % Update image
                this.image = this.image + alphaT.*reshape(colors(k,:,:),1,1,3,this.miniBatchSize);
                this.T = this.T - alphaT;
            end
        end

        function ssim_val = ssimLoss(this)
            % Calculates the structured similarity image metric

            % The diagonal structure of 'window' ensures channel independence.
            mu1 = dlconv(this.image, this.window, 0, 'Padding', 'same');
            mu2 = dlconv(this.image_gt, this.window, 0, 'Padding', 'same');

            mu1_sq = mu1.^2;
            mu2_sq = mu2.^2;
            mu1_mu2 = mu1.*mu2;

            sigma1_sq = dlconv(this.image.^2, this.window, 0, 'Padding', 'same') - mu1_sq;
            sigma2_sq = dlconv(this.image_gt.^2, this.window, 0, 'Padding', 'same') - mu2_sq;
            sigma12   = dlconv(this.image.*this.image_gt, this.window, 0, 'Padding', 'same') - mu1_mu2;

            % Calculate SSIM map
            ssim_map = ((single(2).*mu1_mu2 + this.C1).*(single(2).*sigma12 + this.C2))./...
                ((mu1_sq + mu2_sq + this.C1).*(sigma1_sq + sigma2_sq + this.C2));

            % Average over all dimensions to get scalar
            ssim_val = mean(ssim_map, 'all');
        end


        function pruneAndDensify(this,avgGrad,prunningRatio)
            % Adaptive densification step

            % Sort opacity based on low‑contribution
            % - Low opacity -> barely visible
            % - Low gradient -> not learning
            % - Low product -> dead weight
            [~,pruneIdx] = sort(this.params.alphas_raw.*abs(avgGrad.alphas_raw));

            % Force prunning of gaussians with opacity below threshold
            shouldPrune = this.params.alphas_raw(pruneIdx) < this.forcePruneThreshold;

            % Select gaussians with lowest contribution for prunning
            shouldPrune(1:ceil(prunningRatio*this.numGaussians)) = true;

            % Keep gaussians with opacity above threshold
            shouldPrune = shouldPrune & (this.params.alphas_raw(pruneIdx) < this.forceKeepThreshold);

            % Replace pruned gaussians with cloned or splitted gaussians
            numPrune = sum(shouldPrune);
            if numPrune
                % Select indices to be pruned
                pruneIdx = pruneIdx(shouldPrune);
                
                % Sort based on largest positional gradient
                %  - Large position gradient -> complex shape -> increase density
                [~,cloneIdx] = sort(vecnorm(avgGrad.pws,2,2),'descend');
                
                % Remove indices that are to be pruned
                cloneIdx = setdiff(extractdata(cloneIdx),extractdata(pruneIdx),'stable');
                
                % Select indices to cloned
                cloneIdx = cloneIdx(1:numPrune);

                % Identify candidates for split
                shouldSplit = max(exp(this.params.scales_raw),[],2) > this.scaleThresh*this.sceneScale;

                % Apply densification
                this.params.pws(pruneIdx,:) = this.params.pws(cloneIdx,:);
                this.params.alphas_raw(pruneIdx,:) = this.params.alphas_raw(cloneIdx,:);
                this.params.scales_raw(pruneIdx,:) = this.params.scales_raw(cloneIdx,:) - shouldSplit(cloneIdx,:).*this.splitScaleFactor;
                this.params.scales_raw(cloneIdx,:) = this.params.scales_raw(cloneIdx,:) - shouldSplit(cloneIdx,:).*this.splitScaleFactor;
                this.params.rots_raw(pruneIdx,:) = this.params.rots_raw(cloneIdx,:);
                this.params.shs(pruneIdx,:) = this.params.shs(cloneIdx,:);

                fprintf('Densification: %d clones, %d splits. \n', ...
                    sum(~shouldSplit(cloneIdx,:)), sum(shouldSplit(cloneIdx,:)));
            end
        end

        function saveGaussians(this,filename)
            % Save params to mat file
            params = gather(this.params);
            save(filename, "params");
        end
    end


    methods (Static, Access=private)
        function paramStruct = createLearnableParams(gaussians)
            % Convert struct of doubles to struct of dlarrays with gradients enabled
            paramStruct.pws = dlarray(single(gaussians.pws));
            paramStruct.shs = dlarray(single(gaussians.shs));

            % Raw parameters conversion
            % Scale: log domain
            paramStruct.scales_raw = dlarray(single(log(gaussians.scales + 1e-6)));

            % Alpha: logit domain (inverse sigmoid)
            a = gaussians.alphas;
            a = max(min(a, 0.99), 0.01);
            paramStruct.alphas_raw = dlarray(single(log(a./(1-a))));

            % Rot: pass through
            paramStruct.rots_raw = dlarray(single(gaussians.rots));
        end

        function window_kernel = createWindow(window_size, channel)
            % 1D Gaussian
            sigma = single(1.5);
            coords = 0:single(window_size-1);
            center = floor(window_size/2);
            gauss = exp(-(coords - center).^2/(2*sigma^2));
            gauss = gauss/sum(gauss); % Normalize

            % 2D Gaussian (Outer Product)
            gauss_1d = gauss(:); % Column vector
            gauss_2d = gauss_1d*gauss_1d';

            % Create Weights
            window_kernel = zeros(window_size, window_size, channel, channel, 'single');
            for c = 1:channel
                window_kernel(:, :, c, c) = gauss_2d;
            end

            % Wrap as dlarray with format 'SSCU' (Spatial, Spatial, Channel_In, Channel_Out/Filter)
            window_kernel = dlarray(window_kernel, 'SSCU');
        end
    end
end

