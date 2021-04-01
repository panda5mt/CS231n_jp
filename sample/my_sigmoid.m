%% Sigmoid関数による活性化関数
classdef my_sigmoid
    properties
        out % out = Sigmoid(x)
    end
    methods
%         function obj = my_sigmoid(m)
%             if nargin == 1
%                 
%                 obj.Value = z;
%             end
%         end
        function out = forward(obj, x)
            obj.out = 1 ./ (1+exp(-x));
            out = obj.out;
        end
        
        function dx = backward(obj, dout)
            dx = dout .* (1.0 - obj.out) .* obj.out;
        end
    end
end


