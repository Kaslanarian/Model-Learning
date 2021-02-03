times = 300; % 蒙特卡洛次数
R = zeros(times, 1); %储存u和x1的相关系数
K = zeros(times, 1); %储存遗漏了x2之后，只用y对x1回归的回归系数

for i = 1:times
    n = 30;
    x = -10 + rand(n, 1) * 20;
    u1 = normrnd(0, 5, n, 1) - rand(n, 1);
    x2 = 0.3 * x1 + u1
    u = normrnd(0, 1, n, 1);
    y = 0.5 + 2 * x1 + 5 * x2 + u;
    k = (n * sum(x .* y) - sum(x1) * sum(y)) / (n * sum(x1 .* x1) - sum(x1) * sum(x1));
    K(i) = k;
    u = x2 + u
    r = corrcoef(x1, x2);
    R(i) = r(2, 1);
end

plot(R, K, "*");
xlabel('x_1和u的相关系数');
ylabel("k的估计值")
