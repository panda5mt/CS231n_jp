%% ニューラルネットワークテスト
close all; 
clc;

%% パラメータ初期化
N = 100;    % 各分類ごとのポイント数
D = 2;      % 次元
K = 3;      % 分類数

X = zeros(N*K, D);  % データ格納用行列
y = zeros(N*K, 1);  % 分類ラベル(教師ラベル)

num_examples = size(X, 1);

%% パラメータの初期化
h = 100; % 隠れ層サイズ
W = 0.01 * randn(D, h);
b = zeros(1, h);
W2 = 0.01 * randn(h, K);
b2 = zeros(1, K);

% ハイパーパラメータの設定
step_size = 1;
reg = 1e-3; % 正則化強度


%% スパイラルの行列を作成
for j=0:(K-1)
    ix = (N*j:N*(j+1)-1)+1;
    r = linspace(0.0,1,N); % radius
    t = linspace(j*4, (j+1)*4, N) + (randn(N,1) * 0.2)'; % theta
    
    X(ix,1) = r.*sin(t);
    X(ix,2) = r.*cos(t);
    y(ix,1) = j + 1;
end

% 分類ごとに色を変えたいのでone-hot表記にする
% 特にNNに関係する演算ではない
y_color = y==1:3;

%% 散布図描画
figure(1);
scatter(X(:,1),X(:,2),[],y_color,'filled','MarkerEdgeColor',[0 0 0],'LineWidth',1.5);

%% 勾配降下計算ループ
for i = 1:10000
    
    %% 分類スコアの評価
    M = X * W + b;
    hidden_layer = M .* (0 < M); % ReLU:活性化関数
    scores = hidden_layer * W2 + b2;
    
    % 分類性能計算
    exp_scores = exp(scores);
    probs = exp_scores ./ sum(exp_scores, 2); %[N x K]
    
    
    % 損失関数計算
    correct_logprobs = zeros(num_examples,1);
    
    for zz=1:num_examples
       correct_logprobs(zz,1) = -log(probs(zz, y(zz)));
    end
    
    data_loss = sum(correct_logprobs,'all') / num_examples;
    
    reg_loss = 0.5 * reg * sum (W .^ 2,'all') + 0.5 * reg * sum (W2 .^ 2,'all');
    loss = data_loss + reg_loss;
    
    %% 1000回ごとにどのくらい損失関数が改善したか表示する
    if 0 == mod(i, 1000) || (1 == i)
        Str1 = ['iteration ',num2str(i),': loss ',num2str(loss),'.'];
        disp(Str1); 
    end
    
    dscores = probs;
    
    for zz=1:num_examples
        dscores(zz,y(zz)) = dscores(zz,y(zz)) - 1;
    end
    
    dscores = dscores ./ num_examples;
    
    dW2 = hidden_layer' * dscores;
    db2 = sum(dscores, 1);
    
    dhidden = dscores * W2';
    dhidden = dhidden .* (hidden_layer > 0);
    
    dW = X' * dhidden;
    db = sum(dhidden, 1);
    
    dW2 = dW2 + reg .* W2;
    dW = dW + reg .* W;
    
    W = W - step_size * dW;
    b = b - step_size * db;
    W2 = W2 - step_size * dW2;
    b2 = b2 - step_size * db2;
    
    
end
%% 訓練終わり
M = X * W + b;
hidden_layer = M .* (0 < M); % ReLU:活性化関数

scores = hidden_layer * W2 + b2;

[~,predicted_class] = max(scores,[],2);

% 正解している数
pc_num = sum(predicted_class == y,'all');
% 平均値(%表示)
pc_mean = pc_num / num_examples * 100;
Str2 = ['training accuracy: ',num2str(pc_mean),'%.'];
disp(Str2);

% サンプルデータを入れてみる
for lp1=1:30
    DX = zeros(N*K, D);  % データ格納用行列
    sample_x = 2*rand([1 num_examples])-1;
    sample_y = 2*rand([1 num_examples])-1;

    DX(:,1)= sample_x;
    DX(:,2)= sample_y;


    M = DX * W + b;
    hidden_layer = M .* (0 < M); % ReLU:活性化関数
    scores = hidden_layer * W2 + b2;
    [~,argmax_scores] = max(scores,[],2);

   
    % one-hot表記へ変換
    y_color = argmax_scores==1:3;
    hold on
    figure(1);scatter(DX(:,1),DX(:,2),[],y_color);
    hold off
end




