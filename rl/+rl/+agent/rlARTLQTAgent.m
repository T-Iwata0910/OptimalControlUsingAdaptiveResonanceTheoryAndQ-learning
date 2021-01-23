classdef rlARTLQTAgent < rl.agent.AbstractAgent
    properties (Dependent)
        AgentOptions
    end
    
    properties
        Critic
        
        % NoiseModel
        NoiseModel
        
        Lambda (1, 1) double {mustBeInteger, mustBePositive} = 100
        Rho (1, 1) double {mustBeInteger, mustBePositive} = 20
        Kappa (1, 1) double {mustBeInteger, mustBePositive} = 20
        Nu (1, 1) double {mustBeFinite, mustBePositive} = 0.78  % TODO: Serch in jurnal
        DeltaX (1, 1) double {mustBeInteger, mustBePositive} = 4
        DeltaU (1, 1) double {mustBeInteger, mustBePositive} = 4
        Alpha (1, 1) double {mustBeFinite, mustBePositive} = 0.9  % TODO: Serch in jurnal
        W
        KBuffer (1, 1) DataLogger
    end
    
    properties (Access = private)
        AgentOptions_ = []
        
        % Circular buffer
        ExperienceBuffer
        
        % 1イテレーションあたりのステップ数（この数で一度方策の更新を行う）
        StepNumPerIteration        
        
        % Stop learning value
        % ゲインの更新幅がこの値以下になったら学習を終了する
        StopExplorationValue;
        StopExplorationFlg = false;
        
        SaveExperiences
        
        % Save buffer
        dx
        du
        Tau
        F
        
        %
        v = 1
    end
    methods
        function this = rlARTLQTAgent(varargin)
            %input parser
            narginchk(2, 3);
            
            this = this@rl.agent.AbstractAgent();
            
            % validate inputs
            % see also: rl.util.parseAgentInputs.m
            % infomation check
            oaInfo = varargin(cellfun(@(x) isa(x, 'rl.util.RLDataSpec'), varargin));
            if numel(oaInfo) ~= 2
                error('Action or obsevation infomation is invalid');
            end
            
            % options check
            UseDefault = false;
            opt = varargin(cellfun(@(x) isa(x, 'rl.option.AgentGeneric'), varargin));
            
            % whole check
            if numel(varargin)~=(numel(oaInfo)+numel(opt))
                error(message('rl:agent:errInvalidAgentInput'));
            end
            
            if isempty(opt)
                opt{1} = rlARTLQTAgentOptions;
                UseDefault = true;
            else
                % check otption is compatible
                if ~isa(opt{1}, 'rl.option.rlARTLQTAgentOptions')
                    error(message('rl:agent:errMismatchedOption'));
                end
            end
            
            % set ActionInfo and ObservationInfo
            this.ObservationInfo = oaInfo{1};
            this.ActionInfo = oaInfo{2};
            
            % set agent option(ノイズモデルのインスタンスでthis.ActionInfoを使用するのでActionInfoの設定を終えてから)
            this.AgentOptions = opt{1};
            
            % Create the critic representation
            this.Critic = [];
            this.Critic{1} = createCritic(this);
        end
        
        function action = getActionWithExploration(this, Observation)
            % Given the current observation, select an action
            action = getAction(this,Observation);
            
            % Add random noise to action
            action = applyNoise(this.NoiseModel, action);
            
            % saturate the actions
            action = saturate(this.ActionInfo, action);
            
        end
        
        function set.AgentOptions(this, NewOptions)
            validateattributes(NewOptions,{'rl.option.rlARTLQTAgentOptions'},{'scalar'},'','AgentOptions');
            
            % check if the experience buffer needs to be rebuild
            rebuildExperienceBuffer = isempty(this.ExperienceBuffer) || ...
                this.AgentOptions_.StepNumPerIteration ~= NewOptions.StepNumPerIteration;
            % check to see if we need to rebuild the noise model
            rebuildNoise = isempty(this.NoiseModel) || ...
                ~isequal(this.AgentOptions_.NoiseOptions,NewOptions.NoiseOptions);
            
            this.AgentOptions_ = NewOptions;
            
            this.SampleTime = NewOptions.SampleTime;
            this.StepNumPerIteration = NewOptions.StepNumPerIteration;
            this.StopExplorationValue = NewOptions.StopExplorationValue;
            this.SaveExperiences = NewOptions.SaveExperiences;
            
            this.Lambda = NewOptions.Lambda;
            this.Rho = NewOptions.Rho;
            this.Kappa = NewOptions.Kappa;
            this.DeltaX = NewOptions.DeltaX;
            this.DeltaU = NewOptions.DeltaU;
            
            % build the experience buffer if necessary
            if rebuildExperienceBuffer
                if isempty(this.ExperienceBuffer)
                    buildBuffer(this);
                else
                    resize(this.ExperienceBuffer,this.AgentOptions_.StepNumPerIteration);
                end
            end
            
            % build the noise model if necessary
            if rebuildNoise
                % extract the noise options
                noiseOpts = this.AgentOptions_.NoiseOptions;

                % create the noise model
                actionDims = {this.ActionInfo.Dimension}';
                this.NoiseModel = rl.util.createNoiseModelFactory(...
                    actionDims,noiseOpts,getSampleTime(this));
            end
        end
        function AgentOptions = get.AgentOptions(this)
            AgentOptions = this.AgentOptions_;
        end
    end
    
    methods(Access = protected)
        function action = learn(this, exp)
            
            % calulate ART
            this.v = ART(this, exp);
            
            % Store experiences
            appendExperience(this, exp);
            
            if this.ExperienceBuffer.Length >= this.StepNumPerIteration
                updatePolicy(this);
            end
            
            if this.SaveExperiences
                append(this.KBuffer, getPolicy(this));
            end
            
            action = getActionWithExploration(this, exp{4});
        end
        
        function resetImpl(this)
            % 学習開始時に1度だけ実行
            
            % 実験データをロギング
            % ※RL toolboxのtrainで学習した時には使用することができない(途中で捨てられる)
            if this.SaveExperiences
                attachLogger(this, this.MaxSteps);
                this.KBuffer = DataLogger(this.MaxSteps, "double");
            end
            
            % reset the noise model
            reset(this.NoiseModel);
            
            % Initialize ART parameters TODO: consider
            oDim = this.ObservationInfo.Dimension(1);
            aDim = this.ActionInfo.Dimension(1);
            this.W = rand(oDim*(this.DeltaX+1)+aDim*(this.DeltaU), 1);  % Consider initial module num(now: 1)
            this.dx = [];
            this.du = [];
            this.Tau = zeros(1); % Consider initial module num(now: 1)
            this.F = zeros(1) + this.Lambda; % Consider initial module num(now: 1)
            
            this.StopExplorationFlg = false;
        end
        
        function action = getActionImpl(this, Observation)
            K = getPolicy(this);
            
            action = -K * Observation{:};
        end
        
        function HasState = hasStateImpl(this)
        end
        function setLearnableParametersImpl(this, p)
        end
        function p = getLearnableParametersImpl(this)
        end
        function trainingOptions = validateAgentTrainingCompatibilityImpl(this, trainingOptions)
            if ~strcmpi(trainingOptions.Parallelization,'none')
                % currently do not support parallel training
                error(message('rl:general:errParallelTrainNotSupport'));
            end
        end
        
        function v = ART(this, exp)
            v = zeros(1, length(this.Critic)) + 1 / length(this.Critic);
            x = exp{1}{:};
            u = exp{2}{:};
            dx = exp{4}{:};
            
            % Append queue (According equation(10))
            oDim = this.ObservationInfo.Dimension(1);
            aDim = this.ActionInfo.Dimension(1);
            
            if isempty(this.dx) % to set initial obsrvation
                this.dx = [x; dx];
            else
                this.dx = [this.dx; dx];
            end
            this.du = [this.du; u];
            
            % reshape
            this.dx = this.dx(max(0, length(this.dx) - oDim*(this.DeltaX+1))+1:end);
            this.du = this.du(max(0, length(this.du) - aDim*this.DeltaU)+1:end);
            d = [this.dx; this.du];
            
            if length(d) == oDim*(this.DeltaX+1) + aDim*this.DeltaU
                d = d / norm(d);
                
                this.W(:, this.F==0) = [];  % remove colum (c.f. table 1 l.8)
                this.Tau(this.F==0) = [];
                this.F(this.F==0) = [];
                this.Critic(this.F==0) = [];
                
                v  = this.W' * d / norm(this.W' * d);  % (c.f. table 1 l.9)
                
                [~, j] = sort(v, "descend");  % (c.f. table 1 l.10)
                
                l = 1;  % (c.f. table 1 l.11)
                while (1)
                    jStar = j(l);    % (c.f. table 1 l.12)
                    if this.Tau(jStar) == 0
                        dStar = this.W(:, jStar);
                        if norm(d-dStar) < this.Nu
                            this.W(:, jStar) = this.Alpha * d + (1-this.Alpha)*this.W(:, jStar);  % (c.f. table 1 l.17)
                            this.F(jStar) = this.Lambda;  % (c.f. table 1 l.17)
                            for k = 1 : l - 1
                                this.Tau(j(k)) = this.Rho;  % (c.f. table 1 l.18)
                            end
                            break;
                        end
                    end
                    %% (3)
                    l = l + 1;
                    if l >= this.Kappa || l > length(j)
                        this.W(:, end+1) = d;
                        this.Tau(end+1) = 0;
                        this.F(end+1) = this.Lambda;
                        this.Critic{end+1} = createCritic(this);
                        break;
                    end
                end
                
                this.Tau = max(this.Tau - 1, 0);
                this.F = max(this.F - 1, 0);
            end
        end
        
        function updatePolicy(this)
            gamma = this.AgentOptions.DiscountFactor;
            oaDim = this.ObservationInfo.Dimension(1) + this.ActionInfo.Dimension(1);
            yBuf = zeros(this.ExperienceBuffer.Length,1);
            hBuf = zeros(this.ExperienceBuffer.Length,0.5*oaDim*(oaDim+1));
            minibatch = this.ExperienceBuffer.getLastNData(this.StepNumPerIteration);
            K = getPolicy(this);
            for i = 1 : this.ExperienceBuffer.Length
                % Parse the experience input
                x = minibatch{i}{1}{1};
                u = minibatch{i}{2}{1};
                r = minibatch{i}{3};
                dx = minibatch{i}{4}{1};

                % In the linear case, critic evaluated at (x,u) is Q1 = theta'*h1,
                % critic evaluated at (dx,-K*dx) is Q2 = theta'*h2. The target
                % is to obtain theta such that Q1 - gamma*Q2 = y, that is,
                % theta'*H = y. Following is the least square solution.
                h1 = computeQuadraticBasis(x, u, oaDim);
                h2 = computeQuadraticBasis(dx, -K*dx, oaDim);  %TODO:consider action

                H = h1 - gamma* h2;

                yBuf(i, 1) = r;
                hBuf(i, :) = H;
            end

            % Update the critic parameters based on the batch of
            % experiences
            theta = pinv(hBuf)*yBuf;
            for i = 1 : length(this.v)
                this.Critic{i} = setLearnableParameterValues(this.Critic{i}, {this.v(i) * theta});
            end

            % Reset the experience buffers
            this.ExperienceBuffer.reset();
        end
        
        function k = getPolicy(this)
            theta = getLearnableParameterValues(this.Critic{1});
            w = zeros(size(theta));
            for i = 1 : length(this.v)
                theta = getLearnableParameterValues(this.Critic{i});
                w = w + this.v(i) * theta{:};
            end
            if isa(w, "dlarray")
               w = w.extractdata;
            end
            
            observeDim = this.ObservationInfo.Dimension(1);
            actionDim = this.ActionInfo.Dimension(1);
            n = observeDim+actionDim;
            idx = 1;
            for r = 1:n
                for c = r:n
                    Phat(r,c) = w(idx);
                    idx = idx + 1;
                end
            end
            H  = 1/2*(Phat+Phat');
            Huu = H(observeDim+1:end,observeDim+1:end);
            Hux = H(observeDim+1:end,1:observeDim);
            if rank(Huu) == actionDim
                k = Huu\Hux;
            else
                k = this.K;
            end
        end
       
        function critic = createCritic(this)
            observeDim = this.ObservationInfo.Dimension(1);
            actionDim = this.ActionInfo.Dimension(1);
            n = observeDim+actionDim;
            w0 = 0.1*ones(0.5*(n+1)*n,1);
            
            if verLessThan('rl', '1.2')
                critic = rlRepresentation(@(x,u) computeQuadraticBasis(x,u,n),w0,...
                    {this.ObservationInfo,this.ActionInfo});
            else
                critic = rlQValueRepresentation({@(x,u) computeQuadraticBasis(x,u,n),w0},...
                    this.ObservationInfo,this.ActionInfo);
            end
            critic.Options.GradientThreshold = 1;
%             critic = critic.setLoss('mse');
        end
        
    end
    
    methods(Hidden)
        function appendExperience(this,experiences)
            % append experiences to buffer
            append(this.ExperienceBuffer,{experiences});
        end
    end
    
    methods(Access= private)
        function buildBuffer(this)
            this.ExperienceBuffer = rl.util.ExperienceBuffer(...
                this.AgentOptions_.StepNumPerIteration, ...
                this.ObservationInfo, ...
                this.ActionInfo);
        end
    end
end

%% local function
function B = computeQuadraticBasis(x,u,n)
z = cat(1,x,u);
idx = 1;
for r = 1:n
    for c = r:n
        if idx == 1
            B = z(r)*z(c);
        else
            B = cat(1,B,z(r)*z(c));
        end
        idx = idx + 1;
    end
end
end

function Data = saturate(oaInfo, Data)
    % SATURATE saturates the data based on rlNumericData spec 
    % LowerLimit and UpperLimit. 
    %
    %  DATA = saturate(NUMERICSPEC, DATA)
    %   - If NUMERICSPEC is a scalar, DATA can be numeric or a cell
    %   - If NUMERICSPEC is a vector, DATA should be a cell array
    %   with the same size as NUMERICSPEC.
    %   - Extra elements in DATA will be truncated.

    try
        if iscell(Data)
            for i = 1:numel(oaInfo)
                Data{i} = min(max(Data{i}, oaInfo(i).LowerLimit), oaInfo(i).UpperLimit);
            end
        else
            Data = min(max(Data, oaInfo.LowerLimit), oaInfo.UpperLimit);
        end
%         Data = Data(1:numel(oaInfo));  % truncate extra elements
    catch
        validateattributes(Data, {'numeric','cell'}, {'nonempty'}, '', 'Data');
        if iscell(Data)
            validateattributes(Data, {'cell'}, {'vector','nonempty'}, '', 'Data');
            if (numel(Data) ~= numel(oaInfo))
                error(message('rl:general:errNumericSaturateMismatchNumel'))
            end
        else
            if numel(oaInfo) ~= 1
                error(message('rl:general:errNumericSaturateMismatchNumel'))
            end
        end
    end
end