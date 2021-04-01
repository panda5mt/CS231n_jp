%% Rectified Linear Unit関数による活性化関数
classdef my_relu
    properties
        mask % 行列中の要素で0以下を[1],0より大きいなら[0]にする
    end
    methods
%         function obj = my_relu(m)
%             if nargin == 1
%                 
%                 obj.Value = z;
%             end
%         end
        function out = forward(obj, x)
            obj.mask = (x <= 0);
            out = x .* ~(obj.mask);
        end
        
        function dx = backward(obj, dout)
            dx = dout .* ~(obj.mask);
        end
    end
end
