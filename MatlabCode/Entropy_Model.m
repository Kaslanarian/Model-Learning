function [W] = EntropMethod(Z)
    % EntropyMEthod - 熵权法
    % 输入n*m的矩阵，已经过标准化和正则化
    % 输出 W:熵权，m*1行向量
    [n, m] = size(Z)
    D = zeros(1, m)

    for i = 1:m
        x = Z(:.i);
        p = x / sum(x);
        e = -sum(p .* log(p)) / log(n)%信息熵
        d = 1 - e;
    end

    W = D ./ sum(D)
end
