classdef GaussianSplatter < handle
    % GaussianSplattingTrainer
    % MATLAB implementation of 3D Gaussian Splatting training loop.

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

        % Loss Weights
        lambda = single(0.2);

        % SSIM Constants
        windowSize = single(11);
        C1 = single(0.01^2);
        C2 = single(0.03^2);

        % Constants
        shToColor = single([0.28209479177387814;...
                            0.4886025119029199;...
                            0.4886025119029199;...
                            0.4886025119029199;...
                            1.0925484305920792;...
                            1.0925484305920792;...
                            1.0925484305920792;...
                            0.31539156525252005;...
                            0.5462742152960396]);

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
        window
        X
        Y
        XY
        T
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

            [Xi, Yi] = meshgrid(1:uint32(this.imageWidth), 1:uint32(this.imageHeight));
            this.X = dlarray(repmat(single(Xi),1,1,1),'SSC');
            this.Y = dlarray(repmat(single(Yi),1,1,1),'SSC');
            this.T = dlarray(ones(this.imageHeight,this.imageWidth,1,'single'), 'SSC');
            this.shToColor = dlarray(this.shToColor,'SSC');

            % Move to GPU if input is on GPU
            if isgpuarray(this.camera.id)
                disp('Move arrays to GPU environment...')
                this.params.pws = gpuArray(this.params.pws);
                this.params.shs = gpuArray(this.params.shs);
                this.params.scales_raw = gpuArray(this.params.scales_raw);
                this.params.alphas_raw = gpuArray(this.params.alphas_raw);
                this.params.rots_raw = gpuArray(this.params.rots_raw);

                this.shToColor = gpuArray(this.shToColor);
                this.C1 = gpuArray(this.C1);
                this.C2 = gpuArray(this.C2);
                this.lambda = gpuArray(this.lambda);

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
            this.createImage(params);

            % L1 Loss
            l1_loss = mean(abs(this.image - this.image_gt), 'all');

            % SSIM Loss (Differentiable)
            ssim_loss = 1.0 - this.ssimLoss();

            % Combined Loss
            loss = (1.0 - this.lambda)*l1_loss + this.lambda*ssim_loss;

            % Compute Gradients
            grads = dlgradient(loss, params);
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
            ssim_map = ((single(2.0).*mu1_mu2 + this.C1).*(single(2.0).*sigma12 + this.C2))./...
                ((mu1_sq + mu2_sq + this.C1).*(sigma1_sq + sigma2_sq + this.C2));

            % Average over all dimensions to get scalar
            ssim_val = mean(ssim_map, 'all');
        end

        function createImage(this,params)
            % Create image uisng Gaussian Splatting

            % Strip dimensions
            this.camera = structfun(@stripdims,this.camera,'UniformOutput',false);

            % Reset image
            this.image(:) = 0;

            % Projects Gaussians with Frustum Culling and renders them
            [u,v,alphas,Cov,colors] = this.projectGaussiansWithCulling(params);

            % A simplified Gaussian Splatting Rasterizer
            for b = 1:this.miniBatchSize
                this.T(:) = 1;

                % Compute singular value decomposition
                [U, S, ~] = this.svd3(shiftdim(stripdims(Cov(1:length(alphas),:,:,b)),1));

                % Calculate radii (half-extents)
                radiusPixels = single(3.0).*S; % 3-sigma coverage
                radiusPixels = U(1:2,:,:).*radiusPixels;
                radiusPixels = ceil(extractdata(squeeze(vecnorm(radiusPixels,2,2))))';

                % Compute pixel region of influence
                umin = max(1, floor(extractdata(u(1:length(alphas),:,:,b))) - radiusPixels(:,1,:));
                umax = min(this.imageWidth, ceil(extractdata(u(1:length(alphas),:,:,b))) + radiusPixels(:,1,:));
                vmin = max(1, floor(extractdata(v(1:length(alphas),:,:,b))) - radiusPixels(:,2,:));
                vmax = min(this.imageHeight, ceil(extractdata(v(1:length(alphas),:,:,b))) + radiusPixels(:,2,:));

                % Continue to next when region is not within image
                valid_indices = 1:length(alphas);
                valid_indices = valid_indices((umin <= umax) & (vmin <= vmax));

                for i = 1:length(valid_indices)
                    k = valid_indices(i);

                    % Get bounds
                    umin_k = umin(k); 
                    umax_k = umax(k);
                    vmin_k = vmin(k); 
                    vmax_k = vmax(k);

                    % Gaussian falloff to mask to ignore far pixels
                    alphaT = stripdims([reshape(this.X(vmin_k:vmax_k,umin_k:umax_k,1),[],1) - u(k,:,:,b), reshape(this.Y(vmin_k:vmax_k,umin_k:umax_k,1),[],1) - v(k,:,:,b)])*(U(1:2,:,k)./max(S(:,:,k),1e-6));
                    alphaT = dlarray(reshape(sum(alphaT.*alphaT,2),vmax_k-vmin_k+1,umax_k-umin_k+1),'SSC');
                    alphaT = exp(-single(0.5).*alphaT);
                    alphaT = alphas(k,:,:,b).*alphaT;
                    alphaT = this.T(vmin_k:vmax_k,umin_k:umax_k,1,1).*alphaT;

                    % Update image
                    this.image(vmin_k:vmax_k,umin_k:umax_k,1,b) = this.image(vmin_k:vmax_k,umin_k:umax_k,1,b) + alphaT.*colors(k,:,1,b);
                    this.image(vmin_k:vmax_k,umin_k:umax_k,2,b) = this.image(vmin_k:vmax_k,umin_k:umax_k,2,b) + alphaT.*colors(k,:,2,b);
                    this.image(vmin_k:vmax_k,umin_k:umax_k,3,b) = this.image(vmin_k:vmax_k,umin_k:umax_k,3,b) + alphaT.*colors(k,:,3,b);
                    this.T(vmin_k:vmax_k,umin_k:umax_k,1,1) = this.T(vmin_k:vmax_k,umin_k:umax_k,1,1) - alphaT;

                    % Early termination when fully opaque
                    if max(this.T, [], 'all') < 0.01
                        break;
                    end
                end
            end
        end

        function [u,v,alphas,Cov,colors] = projectGaussiansWithCulling(this,params)
            % Projects Gaussians with Frustum Culling and renders them

            % Project to Camera Space
            gaussians = pagemtimes(repmat(params.pws,1,1,this.miniBatchSize),'none',this.camera.Rcw,'transpose');
            gaussians = gaussians + repmat(pagetranspose(reshape(this.camera.tcw,3,1,this.miniBatchSize)),size(params.pws,1),1,1);

            % Project to Image Plane (Homogeneous division)
            %inv_z = arrayfun(this.fcnInv, gaussians(:,3,:));
            inv_z = single(1.0)./max(gaussians(:,3,:),single(1e-6));
            gaussians(:,1,:) = gaussians(:,1,:).*inv_z.*reshape(this.camera.fx,1,1,this.miniBatchSize) + reshape(this.camera.cx,1,1,this.miniBatchSize);
            gaussians(:,2,:) = gaussians(:,2,:).*inv_z.*reshape(this.camera.fy,1,1,this.miniBatchSize) + reshape(this.camera.cy,1,1,this.miniBatchSize);

            % Frustum Culling
            valid = (gaussians(:,3,:) > single(0.2)) &...
                    (gaussians(:,3,:) < single(100.0)) &...
                    (gaussians(:,1,:) > -single(this.imageWidth)*single(0.5)) &...
                    (gaussians(:,1,:) < single(this.imageWidth)*single(1.5)) & ...
                    (gaussians(:,2,:) > -single(this.imageHeight)*single(0.5)) &...
                    (gaussians(:,2,:) < single(this.imageHeight)*single(1.5));

            % If no guassians splat in 2d image return empty image
            nr_valid = sum(valid,1);
            if ~any(nr_valid)
                u = [];
                v = [];
                alphas = [];
                Cov = [];
                colors = [];
                return;
            end

            % Preallocate outputs
            numValid = extractdata(max(nr_valid));
            u = dlarray(zeros(numValid,1,1,this.miniBatchSize,like=this.shToColor), 'SSCB');
            v = dlarray(zeros(numValid,1,1,this.miniBatchSize,like=this.shToColor), 'SSCB');
            alphas = dlarray(zeros(numValid,1,1,this.miniBatchSize,like=this.shToColor), 'SSCB');
            Cov = dlarray(zeros(numValid,3,3,this.miniBatchSize,like=this.shToColor), 'SSSB');
            colors = dlarray(zeros(numValid,1,3,this.miniBatchSize,like=this.shToColor), 'SSCB');
            Sh = dlarray(zeros(numValid,length(this.shToColor),like=this.shToColor), 'SSC');

            % Sort by depth (Painter's Algorithm)
            for b = 1:this.miniBatchSize
                [~, sortIdx] = sort(gaussians(valid(:,:,b),3,b), 'descend');
                validIdx = find(extractdata(valid(:,:,b)));
                sortIdx = validIdx(sortIdx);
                numIdx = length(sortIdx);
                u(1:numIdx,:,:,b) = gaussians(sortIdx,1,b);
                v(1:numIdx,:,:,b) = gaussians(sortIdx,2,b);

                % Sigmoid
                alphas(1:numIdx,:,:,b) = single(1.0)./(single(1.0) + exp(-params.alphas_raw(sortIdx)));

                % Normalize quaternions
                quat = params.rots_raw(sortIdx,:);
                quat = quat./max(vecnorm(quat, 2, 2),1e-6);

                % Convert quaternions to rotation matrices
                Cov(1:numIdx,1,1,b) = single(1.0)-single(2.0)*(quat(:,3).*quat(:,3) + quat(:,4).*quat(:,4));
                Cov(1:numIdx,1,2,b) = single(2.0)*(quat(:,2).*quat(:,3) - quat(:,4).*quat(:,1));
                Cov(1:numIdx,1,3,b) = single(2.0)*(quat(:,2).*quat(:,4) + quat(:,3).*quat(:,1));
                Cov(1:numIdx,2,1,b) = single(2.0)*(quat(:,2).*quat(:,3) + quat(:,4).*quat(:,1));
                Cov(1:numIdx,2,2,b) = single(1.0)-single(2.0)*(quat(:,2).*quat(:,2) + quat(:,4).*quat(:,4));
                Cov(1:numIdx,2,3,b) = single(2.0)*(quat(:,3).*quat(:,4) - quat(:,2).*quat(:,1));
                Cov(1:numIdx,3,1,b) = single(2.0)*(quat(:,2).*quat(:,4) - quat(:,3).*quat(:,1));
                Cov(1:numIdx,3,2,b) = single(2.0)*(quat(:,3).*quat(:,4) + quat(:,2).*quat(:,1));
                Cov(1:numIdx,3,3,b) = single(1.0)-single(2.0)*(quat(:,2).*quat(:,2) + quat(:,3).*quat(:,3));

                % Combine camera and Gaussian rotations
                Cov(:,:,:,b) = shiftdim(pagemtimes(this.camera.Rcw(:,:,b), shiftdim(stripdims(Cov(:,:,:,b)),1)),2);

                % Apply scaling and map to screen space
                Cov(1:numIdx,:,1,b) = Cov(1:numIdx,:,1,b).*exp(min(max(params.scales_raw(sortIdx,1),single(-10)),single(10))).*inv_z(sortIdx,:,b).*this.camera.fx(b);
                Cov(1:numIdx,:,2,b) = Cov(1:numIdx,:,2,b).*exp(min(max(params.scales_raw(sortIdx,2),single(-10)),single(10))).*inv_z(sortIdx,:,b).*this.camera.fx(b);
                Cov(1:numIdx,:,3,b) = Cov(1:numIdx,:,3,b).*exp(min(max(params.scales_raw(sortIdx,3),single(-10)),single(10))).*inv_z(sortIdx,:,b).*this.camera.fx(b);

                % Normalize positions
                pos = gaussians(sortIdx,:,b);
                pos = pos./max(vecnorm(pos, 2, 2),1e-6);

                % Apply spherical color coefficients
                Sh(1:numIdx,1) = this.shToColor(1);
                Sh(1:numIdx,2) = this.shToColor(2).*(-pos(:,1));
                Sh(1:numIdx,3) = this.shToColor(3).*(-pos(:,2));
                Sh(1:numIdx,4) = this.shToColor(4).*( pos(:,3));
                Sh(1:numIdx,5) = this.shToColor(5).*( pos(:,1).*pos(:,2));
                Sh(1:numIdx,6) = this.shToColor(6).*(-pos(:,1).*pos(:,3));
                Sh(1:numIdx,7) = this.shToColor(7).*(-pos(:,2).*pos(:,3));
                Sh(1:numIdx,8) = this.shToColor(8).*( single(3.0).*pos(:,3).*pos(:,3) - single(1.0));
                Sh(1:numIdx,9) = this.shToColor(9).*( pos(:,1).*pos(:,1) - pos(:,2).*pos(:,2));

                % Map to screen colors
                colors(1:length(sortIdx),:,1,b) = max(min(single(0.5) + sum(Sh(1:numIdx,:).*params.shs(sortIdx,:,1),2),single(1.0)),single(0.0));
                colors(1:length(sortIdx),:,2,b) = max(min(single(0.5) + sum(Sh(1:numIdx,:).*params.shs(sortIdx,:,2),2),single(1.0)),single(0.0));
                colors(1:length(sortIdx),:,3,b) = max(min(single(0.5) + sum(Sh(1:numIdx,:).*params.shs(sortIdx,:,3),2),single(1.0)),single(0.0));
            end
        end

        function pruneAndDensify(this,avgGrad,prunningRatio)
            % Adaptive densification step

            % Sort opacity based on low‑contribution
            % - Low opacity -> barely visible
            % - Low gradient -> not learning
            % - Low product -> dead weight
            [~,pruneIdx] = sort(this.params.alphas_raw.*abs(avgGrad.alphas_raw));

            % Force prunning of gaussians with opacity below threshold
            shouldPrune = (this.params.alphas_raw(pruneIdx) < this.forcePruneThreshold);

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
            paramStruct.shs = dlarray(single(resize(reshape(gaussians.shs,[],1,3),9,dimension=2)));

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

        function [U, S, V] = svd3(A)
            % SVD3 Computes the singular value decomposition of 3x3 matrices.
            %   [U, S, V] = svd3(A) returns the singular value decomposition of a
            %   batch of 3x3 matrices such that A = U * S * V'.
            %
            %   Input:
            %       A - dlarray or single array of size 3-by-3-by-N (or 3x3x... batch)
            %
            %   Outputs:
            %       U - Left singular vectors (3x3xN)
            %       S - Singular values as diagonal matrices (3x3xN)
            %       V - Right singular vectors (3x3xN)
            %
            %   Notes:
            %       - Supports dlarray and dlgradient.
            %       - Uses one-sided Jacobi iterations (approx 5 sweeps).

            % Initialize V as Identity
            batch_size = size(A,3);
            V = dlarray(repmat(eye(3,like=extractdata(A(1))), 1, 1, batch_size));

            % Run Jacobi Iterations
            num_iters = 5; % 5 sweeps is usually sufficient for single precision 3x3
            for iter = 1:num_iters
                % Iterate over pairs (1,2), (1,3), (2,3)
                [A,V] = jacobi_step(A,V,1,2);
                [A,V] = jacobi_step(A,V,1,3);
                [A,V] = jacobi_step(A,V,2,3);
            end

            % Compute Singular Values (Norm of orthogonal columns of A)
            S = sqrt(sum(A.*A, 1));

            % Compute U (Normalized columns of A)
            U = A./max(S,1e-6);

            function [A, V] = jacobi_step(A, V, p, q)
                % Extracts columns p and q
                Ap = A(:,p,:);
                Aq = A(:,q,:);

                % Compute elements of the 2x2 Gram matrix
                a = sum(Ap.*Ap, 1);
                b = sum(Aq.*Aq, 1);
                d = sum(Ap.*Aq, 1);

                % Calculate rotation angle safely.
                numer = single(2.0)*d;
                denom = a-b;
                
                % Mask is 0 if isotropic (risky), 1 otherwise
                is_safe = (abs(numer) > single(1e-6)) | (abs(denom) > single(1e-6));
                
                % If unsafe, replace inputs with dummy values to get valid
                % atan2 gradient, then zero out the angle using the mask.
                safe_numer = numer + single(~is_safe);
                safe_denom = denom + single(~is_safe);

                % Calculate rotation angle to annihilate d
                theta = single(0.5).*atan2(safe_numer, safe_denom);
                theta = theta.*single(is_safe); 
                c = cos(theta);
                s = sin(theta);

                % Broadcast c and s to 1x1xN
                c = reshape(c, 1, 1, []);
                s = reshape(s, 1, 1, []);

                % Apply rotation to A (columns)
                A(:,p,:) =  c.*Ap + s.*Aq;
                A(:,q,:) = -s.*Ap + c.*Aq;

                % Apply rotation to V (columns)
                % Accumulate the rotations
                Vp = V(:,p,:);
                Vq = V(:,q,:);

                V(:,p,:) =  c.*Vp + s.*Vq;
                V(:,q,:) = -s.*Vp + c.*Vq;
            end
        end
    end
end

