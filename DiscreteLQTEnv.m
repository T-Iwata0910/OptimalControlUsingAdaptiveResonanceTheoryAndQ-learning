classdef DiscreteLQTEnv < rl.env.MATLABEnvironment
    %DISCRETELQTENV: Template for defining custom environment in MATLAB.    
    
    %% Properties (set properties' attributes accordingly)
    properties
        % Specify and initialize environment's necessary properties    
        % Augument System
        T;
        B1;
        
        % Augument reward
        Q1;
        R;
    end
    
    properties
        % Initialize system state [x,dx,theta,dtheta]'
        State;
        initState;
    end
    
    properties(Access = protected)
        % Initialize internal flag to indicate episode termination
        IsDone = false        
    end

    %% Necessary Methods
    methods              
        % Contructor method creates an instance of the environment
        % Change class name and constructor name accordingly
        function this = DiscreteLQTEnv(Ad, Bd, Cd, Fd, Q, R, varargin)
            % input parser
            obsDim = size(Ad, 2) + size(Fd, 2);
            checkX0 = @(x) size(x, 1) == obsDim;
            p = inputParser();
            addOptional(p, 'x0', [], checkX0);
            parse(p, varargin{:});
            
            % Initialize Observation settings
            
            ObservationInfo = rlNumericSpec([obsDim 1]);
            ObservationInfo.Name = 'Augument States of system dynamics and references';
            ObservationInfo.Description = 'x, r';
            
            % Initialize Action settings
            actNum = size(Bd, 2);
            ActionInfo = rlNumericSpec([actNum 1]);
            ActionInfo.Name = 'LQT system Action';
            
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            
            % Initialize augument system dynamics
            this.T = blkdiag(Ad, Fd);
            this.B1 = [Bd; zeros(size(Fd, 1), size(Bd, 2))];
            this.State = zeros(size(this.T, 1), 1);
            this.initState = p.Results.x0;
            
            % Initialize reward function parameter
            C1 = [Cd -eye(size(Cd, 1))];
            this.Q1 = C1' * Q * C1;
            this.R = R;
        end
        
        % Apply system dynamics and simulates the environment with the 
        % given action for one step.
        function [Observation,Reward,IsDone,LoggedSignals] = step(this,Action)
            LoggedSignals = [];           
            
            x = this.State;
            % Update system states
            this.State = this.T * x + this.B1 * Action;
            Observation = this.State;
            
            % 
            IsDone = false;
            this.IsDone = IsDone;
            
            % Get reward
            Reward = -x'*this.Q1*x -Action'*this.R*Action;
            
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);
        end
        
        % Reset environment to initial state and output initial observation
        function initialObservation = reset(this)
            if isempty(this.initState)
                initialObservation = rand(size(this.State, 1), 1);
            else
                initialObservation = this.initState;
            end
            this.State = initialObservation;
            
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);
        end
    end
    %% Optional Methods (set methods' attributes accordingly)
    methods               
        % (optional) Visualization method
        function plot(this)
            % Initiate the visualization
            
            % Update the visualization
            envUpdatedCallback(this)
        end
    end
    
    methods (Access = protected)
        % (optional) update visualization everytime the environment is updated 
        % (notifyEnvUpdated is called)
        function envUpdatedCallback(this)
        end
    end
end
