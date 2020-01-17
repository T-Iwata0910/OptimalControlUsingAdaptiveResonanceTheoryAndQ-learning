classdef DiscreteLQTEnv < rl.env.MATLABEnvironment
    %DISCRETELQTENV: Template for defining custom environment in MATLAB.    
    
    %% Properties (set properties' attributes accordingly)
    properties
        % Specify and initialize environment's necessary properties    
%         % System Dynamics
%         A;
%         B;
%         C;
%         
%         % Reference Dynamics
%         F;
%         
        % Augument System
        T;
        B1;
        
%         % Reward Dynamics
%         Q;
%         R;
        
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
        function this = DiscreteLQTEnv(Ad, Bd, Cd, Fd, Q, R, x0)
            
            
            % Initialize Observation settings
            obsDim = size(Ad, 2) + size(Fd, 2);
            ObservationInfo = rlNumericSpec([obsDim 1]);
            ObservationInfo.Name = 'Augument States of system dynamics and references';
            ObservationInfo.Description = 'x, r';
            
            % Initialize Action settings
            actNum = size(Bd, 2);
            ActionInfo = rlNumericSpec([actNum 1]);
            ActionInfo.Name = 'LQT system Action';
            
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            
            % Initialize Dynamics of System
%             this.A = A;
%             this.B = B;
%             this.C = C;
%             this.F = F;
%             this.Q = Q;
%             this.R = R;
            
            % Initialize augument system dynamics
            this.T = blkdiag(Ad, Fd);
            this.B1 = [Bd; zeros(size(Fd, 1), size(Bd, 2))];
            this.State = zeros(size(this.T, 1), 1);
            this.initState = x0;
            
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
        function InitialObservation = reset(this)
            InitialObservation = this.initState;
            this.State = InitialObservation;
            
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
        
        % (optional) Properties validation through set methods
%         function set.State(this,state)
%             validateattributes(state,{'numeric'},{'finite','real','vector','numel',4},'','State');
%             this.State = double(state(:));
%             notifyEnvUpdated(this);
%         end
%         function set.Ts(this,val)
%             validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','Ts');
%             this.Ts = val;
%         end
    end
    
    methods (Access = protected)
        % (optional) update visualization everytime the environment is updated 
        % (notifyEnvUpdated is called)
        function envUpdatedCallback(this)
        end
    end
end
