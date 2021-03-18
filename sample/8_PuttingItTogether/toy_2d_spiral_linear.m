%% 線形分類テスト
close all; 
clc;

N = 100;    % 各分類ごとのポイント数
D = 2;      % 次元
K = 3;      % 分類数

X = zeros(N*K, D);  % データ格納用行列
y = zeros(N*K, 1);  % 分類ラベル(教師ラベル)

num_examples = size(X, 1);

%% 線形分類器の訓練
W = 0.01 * randn(D, K);
b = zeros(1, K);

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
for i = 1:200
    scores = X * W + b; %　分類スコアの評価
    exp_scores = exp(scores);
    probs = exp_scores ./ sum(exp_scores, 2); %[N x K]
    
    % 損失関数計算
    correct_logprobs = zeros(num_examples,1);
    
    for zz=1:num_examples
       correct_logprobs(zz,1) = -log(probs(zz, y(zz)));
    end
    
    data_loss = sum(correct_logprobs,'all') / num_examples;
    
    reg_loss = 0.5 * reg * sum (W .^ 2,'all');
    loss = data_loss + reg_loss;
    
    %% 10回ごとにどのくらい損失関数が改善したか表示する
    if 0 == mod(i, 10) || (1 == i)
        Str1 = ['iteration ',num2str(i),': loss ',num2str(loss),'.'];
        disp(Str1); 
    end
    
    dscores = probs;
    
    for zz=1:num_examples
        dscores(zz,y(zz)) = dscores(zz,y(zz)) - 1;
    end
    
    dscores = dscores ./ num_examples;
    
    
    dW = X' * dscores;
    db =  sum (dscores, 1);
    
    dW = dW + reg .* W;
    
    W = W - step_size * dW;
    b = b - step_size * db;
end
%% 訓練終わり
scores = X * W + b;
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


    s_scores = DX * W + b; %　分類スコアの評価
    exp_scores = exp(s_scores);
    probs = exp_scores ./ sum(exp_scores, 2); %[N x K]

    [~,argmax_probs] = max(probs,[],2);

    % one-hot表記へ変換
    y_color = argmax_probs==1:3;
    
    hold on % 先程のグラフに上書きする
    figure(1);scatter(DX(:,1),DX(:,2),[],y_color);
    hold off
end




