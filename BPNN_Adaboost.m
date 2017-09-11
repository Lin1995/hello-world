% 用Adaboost改进BPNN的预测能力，搭建BP_Adaboost强预测器，然后利用预测值进行分类
% last modified on Jun,12th, 2017 by Lin
clear all

load data
ny = 1;
nhl = 2;  % 隐藏层层数
ncl = 10; % 弱预测器的数量

ntr = size(x_train,1);
nte = size(x_test,1);
m = size(x_train,2);

x_train = x_train';   %转置之后行表示变量，列表示观测样本
x_test = x_test';     %转置之后行表示变量，列表示观测样本
y_train = y_train';    %转置之后行表示输出（响应），列表示观测样本
y_test = y_test';      %转置之后行表示输出（响应），列表示观测样本
[inputn, inputps] = mapminmax(x_train);   %inputn是经过归一化后的数据，inputps是归一化过程中的参数（每个变量的均值及标准差）
[outputn, outputps] = mapminmax(y_train);
hiddenLayer = ones(1,nhl);
hiddenLayer = hiddenLayer * 21;   % hiddenLayer表示隐藏层的层数及每层神经元的数量

BPoutput_train = cell(ncl,1);
BPoutput_test = cell(ncl,1);
D = ones(ncl+1,ntr)/ntr;   %训练样本权重，第i行代表第i个预测器（最后一行没有意义），每列代表每个样本
error = zeros(ncl,1);
at = zeros(ncl,1);   %弱预测器的权重

for i=1:ncl
    %弱预测器i的训练集回判
    net = feedforwardnet(hiddenLayer,'traincgp');  % 也可以用net = newff(inputn, outputn, hiddenLayer); 
    [net, tr] = train(net, inputn, outputn, 'Useparallel','yes');
    an = sim(net, inputn, 'Useparallel','yes');   
    BPoutput = mapminmax('reverse', an, outputps);    %将预测结果按照训练集输出响应的参数进行逆归一化处理
    BPoutput_train{i,1} = BPoutput;
    
    %统计错误样本的权重和
    for j=1:ntr
        if max(abs(BPoutput(:,j) - y_train(:,j))) > 0.5   %回判预测编码与实际编码有任意一位相差0.5以上，表示会误判
            error(i,1) = error(i,1) + D(i,j);
            D(i+1,j) = D(i,j) * 1.5;   % 增大误判样本的权重，乘以多少自己确定（如这里取1.5）
        else
            D(i+1,j) = D(i,j);
        end
    end
    
    %弱预测器i权重
%     at(i) = 0.5 * log((1-error(i,1))/error(i,1));
    at(i) = 0.5 / exp(abs(error(i)));
    D(i+1,:) = D(i+1,:) / sum(D(i+1,:));  %样本权重归一化
    
    %弱预测器i的预测结果
    inputn_test = mapminmax('apply', x_test, inputps);  %将预测集按照训练集输入变量的参数进行归一化处理
    an = sim(net, inputn_test);
    BPoutput = mapminmax('reverse', an, outputps);    %将预测结果按照训练集输出响应的参数进行逆归一化处理
    BPoutput_test{i,1} = BPoutput;
    
end

%若预测器权重归一化
at = at / sum(at);

output = zeros(ny, nte);
for i=1:ncl
    output = output + at(i,1) * BPoutput_test{i,1};
end
output = round(output);
ncor = 0;    % number of correct predictions
for i=1:nte
    if isequal(output(:,i),y_test(:,i))
        ncor = ncor + 1;
    end
end
accuracy_pred = ncor / nte * 100;
fprintf(1,'预测的准确率是： %4.2f%% \n', accuracy_pred);

output = zeros(ny, ntr);
for i=1:ncl
    output = output + at(i,1) * BPoutput_train{i,1};
end
output = round(output);
ncor = 0;    
for i=1:ntr
    if isequal(output(:,i),y_train(:,i))
        ncor = ncor + 1;
    end
end
accuracy_return = ncor / ntr * 100;
fprintf(1,'回判的准确率是： %4.2f%% \n', accuracy_return);
