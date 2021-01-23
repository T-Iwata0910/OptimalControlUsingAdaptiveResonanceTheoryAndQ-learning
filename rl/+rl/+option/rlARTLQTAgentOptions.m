classdef rlARTLQTAgentOptions < rl.option.rlLQTAgentOptions
    
    properties
        Lambda (1, 1) double {mustBeInteger, mustBePositive} = 100
        Rho (1, 1) double {mustBeInteger, mustBePositive} = 20
        Kappa (1, 1) double {mustBeInteger, mustBePositive} = 20
        DeltaX (1, 1) double {mustBeInteger, mustBePositive} = 4
        DeltaU (1, 1) double {mustBeInteger, mustBePositive} = 4
    end
    
    methods
        function this = rlARTLQTAgentOptions(varargin)
            this = this@rl.option.rlLQTAgentOptions(varargin{:});
            
            parser = this.Parser;
            
            addParameter(parser, "Lambda", 100);
            addParameter(parser, "Rho", 20);
            addParameter(parser, "Kappa", 20);
            addParameter(parser, "DeltaX", 4);
            addParameter(parser, "DeltaU", 4);
            
            parse(parser, varargin{:});
            this.Parser = parser;
            this.Lambda = parser.Results.Lambda;
            this.Rho = parser.Results.Rho;
            this.Kappa = parser.Results.Kappa;
            this.DeltaX = parser.Results.DeltaX;
            this.DeltaU = parser.Results.DeltaU;
        end
    end
    
end