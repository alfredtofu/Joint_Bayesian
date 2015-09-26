function mapping = JointBayesian(X, labels)
% X: 特征n*m，每行是一个样本
% labels: n*1
% 

% (C) Laurens van der Maaten, Delft University of Technology
% Make sure data is zero mean
%     mapping.mean = mean(X, 1);
%     [COEFF,SCORE] = princomp(X,'econ');
%     X = SCORE(:,1:400);
%     X = bsxfun(@minus,X,mapping.mean);
sample_count = length(labels);
fea_dims = size(X,2);

% Make sure labels are nice
[classes, ~, labels] = unique(labels);
class_count = length(classes);

samples_per_class = cell(class_count, 1);
% 所有的Variants的数量
%   注意，这是所有类别的Variants的数量加起来，
%   如果某个类别只有一个样本，那么variations为0）
variant_count = 0;

% 只是一个标记，后面在算F跟G的时候，样本数一样的subject，他们的F跟G是相等的，
% 使用这个标记可以减少重复计算
count2ind = containers.Map('KeyType','double','ValueType','int32');

for i = 1:class_count
  % Get all instances with class i
  samples_per_class{i} = X(labels == i, :);
  count = size(samples_per_class{i}, 1);
  if count > 1
    variant_count = variant_count + count;
  end;
  if ~isKey(count2ind, count)
    count2ind(count) = count2ind.Count + 1;
  end;
end;
fprintf('Class count: %d\n', class_count);
fprintf('Variants count: %d\n', variant_count);

tic;
% 同一类的所有样本的均值
u = zeros(fea_dims, class_count);
% 第二维是variant_count，
% 因为算within-class，所以需要错位
% 某些列只有特定的类别才有值存在，否则为0
% 注意的是，在填值的时候，需要先减均值，
% 这样在算协方差的时候就可以直接点乘，否则0就没意义了。
ep = zeros(fea_dims, variant_count);
cur_variant_ind = 1;
% Sum over classes
for i = 1:class_count
  % Update within-class scatter
  u(:,i) = mean(samples_per_class{i}, 1)';

  count = size(samples_per_class{i}, 1);
  if count > 1
    ep(:, cur_variant_ind:cur_variant_ind + count - 1) = ...
      bsxfun(@minus,samples_per_class{i}', u(:,i));
    cur_variant_ind = cur_variant_ind + count;
    %             C = cov(cur{i});
    %             p = size(cur{i}, 1) / withinCount;
    %             Sw = Sw + (p * C);
  end;
end;

% 下面是对应不同的初始化方法
Su = cov(u'); % identify的协方差
Sw = cov(ep'); % variant的协方差
%     Su = u*u'/nc;
%     Sw = ep*ep'/withinCount;

%     Su = cov(rand(n,n));
%     Sw = cov(rand(5*n,n));
toc;
%     F = inv(Sw);
%     mapping.Su = Su;
%     mapping.Sw = Sw;
%     mapping.G = -1 .* (2 * Su + Sw) \ Su / Sw;
%     mapping.A = inv(Su + Sw) - (F + mapping.G);
%     mappedX = X;
% end

oldSw = Sw;
SuFG = cell(count2ind.Count, 1);
SwG = cell(count2ind.Count, 1);

% 终止条件有两个
% (1) 达到最大迭代次数
% (2) 协方差变化少于某个阈值
max_iter = 500;
epsilon = 1e-6;
keys_of_count2ind = keys(count2ind);
for iter=1:max_iter
  %         tic;
  F = inv(Sw);
  ep = zeros(fea_dims, sample_count);
  cur_variant_ind = 1;
  for g = 1:numel(keys_of_count2ind);
    count = keys_of_count2ind{g};
    ind = count2ind(count);
    G = -1 .* (count .* Su + Sw) \ Su / Sw;
    SuFG{ind} = Su * (F + count .* G);
    SwG{ind} = Sw * G;
  end;
  
  for i = 1:class_count
    count = size(samples_per_class{i}, 1);
    ind = count2ind(count);
    u(:, i) = sum(SuFG{ind} * samples_per_class{i}', 2);
    ep(:, cur_variant_ind:cur_variant_ind + count -1) = ...
      bsxfun(@plus, samples_per_class{i}', ...
        sum(SwG{ind} * samples_per_class{i}', 2));
      
    cur_variant_ind = cur_variant_ind + count;
  end;
  
  Su = cov(u');
  Sw = cov(ep');
  %     Su = u*u'/nc;
  %     Sw = ep*ep'/withinCount;
  incre_of_Sw = norm(Sw - oldSw) / norm(Sw);
  fprintf('iter %d: diff %f\n', iter, incre_of_Sw);
  %         toc;
  if incre_of_Sw < epsilon
    break;
  end;
  oldSw = Sw;
end;

F = inv(Sw);
mapping.G = -1 .* (2 * Su + Sw) \ Su / Sw;
mapping.A = inv(Su + Sw) - (F + mapping.G);
mapping.Sw = Sw;
mapping.Su = Su;
%     mapping.U = chol(-G,'upper');
%     mapping.COEFF = COEFF;
%     mapping.y = mapping.U * X';

