```matlab
isprime
mod, rem
fix, round, ceil, floor;
>>> format long # 改变输出不改变输入
ans, i, j, pi, NaN
who, whos
find(logical) # 查找元素
```

```matlab
>>>save mydata var1 var2 
>>>load mydata # 可直接使用var1, var2,或者直接load("mydata.mat")
```

```matlab
A = B + i*C
```

```matlab
begin:step:end #冒号表达式，可省略step
linspace(a, b, n)
```

```matlab
A = {10, 'A', [1; 2], 11, 'B', [3; 4]} #单元矩阵
A(a, b) #	第a行第b列，越界赋值扩展
A(a) #内存中第a+1个元素（按列排序）
```

```matlab
sub2ind(S, I, J) # 将对应位置的元素转换成序号
>>> A=[1:3;4:6]
>>> D = sub2ind(size(A), [1,2;2,2], [1,1;3,2])
D = 
	1	2 #A(1, 1), A(2, 1)
	6	4
ind2sub(S, D) # 序号转下标
>>> [I, J] = ind2sub([3, 3], [1, 3, 5]) # 3行3列矩阵，第一个，第三个，第五个元素的下标
```

```matlab
# end运算符表示某一维最后一个元素下标
A = [1 2 3;3 4 5]
A(1, end)
```

```matlab
A(:, [2, 4])=[] #第2，4列删除
```

```matlab
reshape(A, m, n)
A(:) # 堆叠成列向量
```

```matlab
# 矩阵左除
B\A = inv(B) * A 
# 矩阵右除
B/A = B * inv(A)

>>>3/4
ans = 0.7500
>>>3\4
ans = 1.333
```

```matlab
# 点运算
A .* B # 对应元素进行运算
```



```matlab
zeros(n), zeros(m, n), zeros(size(A))
ones(n), 
eye(n)
rand(n) # 0~1
randi(n, a, b) % a*b矩阵，元素为1~n随机
randi([m, n], a, b)
randn(n) # 均值为0，方差为1的正态分布
diag(A) # 提取主对角线元素
diag(A, k)
triu(A) # A的主对角线及以上的元素
triu(A, k)
tril(A)
tril(A, k)
A.' # 共轭转置
fliplr(A) # 左右翻转
flipud(A) # 上下翻转
```

```matlab
det(A)
rank(A)
trace(A)
norm(v)
norm(v, 1)
norm(v, inf)
cond(A, 1)
E = eig(A)
[X, D] = eig(A) # 特征向量，特征值

A = sparse(S) # 稀疏存储
S = full(A) #完全存储

A = sparse([1, 2, 2], [2, 1, 4], [4, 5, -7])
B =  fill(A)

A = spconvert([1, 2, 2; 2, 1, 4; 4, 5, -7])
```

<u>条件结构中，若条件是矩阵：则当且仅当矩阵非空且无0元素时为真</u>

```matlab
for index=a:b:c
	TODO()
end
```

匿名函数：

```matlab
>>>f=@(x, y) x^2 + y^2
>>>g=@sin
```

```matlab
nargin # 输入参数
```

全局变量

```matlab
function f=wad(x, y)
	global ALPHA, BETA
	f = ALPHA*x + BETA*y;
end

>>> global ALPHA, BETA
>>> ALPHA=1
>>> BETA=2
>>> s = wad(1, 2)
s = 
	5
```





```matlab
plot(x, y)
# x是向量，y是矩阵
if y.cols == x.len
	x为横坐标，y的每个行向量为纵坐标，条数为y的行数
else if y.rows == x.len
	x为横坐标，y的每个列向量为纵坐标，条数为y的列
	
fplot(f, lims, options) # fplot(@(x) sin(1./x), [0, 0.2], 'b')
fplot(fx, fy, tlims, options) # fplot(@(t) t.*sin(t), @(t) t.*cos(t), [0, 10*pi], 'r')
```

```matlab
semilogx, semilogy, loglog
polar(theta, rho, 选项)
bar(y, style) #sytle : "grouped" or "stacked"
bar(x, y)

hist(y)
hist(y, x)
y = randn(500, 1);
x = -3:0.2:3
hist(y), hist(y, x)

rose(theta, x)
pie(x, explode)
score = [5, 17, 23, 9, 4];
ex = [0, 0, 0, 0, 1];
pie(score, x)

A = [4, 5];
B = [-10, 0];
C = A + B;
hold on;
quiver(0, 0, A(1), A(2));
quiver(0, 0, B(1), B(2));
quiver(0, 0, C(1), C(2));

t = 0:pi/50:6 * pi
x = cos(t)
y = sin(t)
z = 2 * t
plot3(x, y, z)
xlabel('X'), ylabel('Y'), zlabel('Z')
grid on

fplot3(fx, fy, fz, tlims)


# 三维面图：
x = 2:6;
y = (3:8)';
[X, Y] = meshgrid(x, y);
Z = randn(size(X));
plot3(X, Y, Z);
grid on;
mesh(X, Y, Z)
surf(X, Y, Z)
meshc, meshz, surfc, surfl

[X, Y, Z] = sphere(n) # 单位球
[X, Y, Z] = cylinder(R, n) # 半径函数，间隔点
peak

fsurf(fx, fy, fz, [umin, umax, vmin, vmax])
fmesh(fx, fy, fz, [umin, umax, vmin, vmax])

view(az, el) #az 方位角，el,仰角
view(2) #放在平面看

colormap cmapname # 设色图名
```

```matlab
y = max(v) # 复数按模取最大
[y, k] = max(v) # 最大元素及其索引
max(A) # 返回行向量，每个元素是该列最大值
[Y, U] = max(A) # ...及其索引（行号）
max(A, [], dim) # dim = 1:功能等价于max(A), dim=2:返回列向量，每一行是该行最大值
max(A(:)) #堆叠从而求得最大

mean(), median() #平均，中位数
sum(), prod()
cumsum(), cumprod() # 累加和，累乘积

std(X), std(A)
std(A, flag, dim) # flag=0:样本标准差(/n-1)->默认，=1:总体标准差(/n)

corrcoef(A) # 相关系数矩阵，A(i,j)是A第i，j列的相关系数
corrcoef(X, Y) # X，Y是向量算X和Y的相关系数
[R, P] = corrcoef(A) # P是每个相关系数的p值

sort(X) #升序排序
[Y, I] = sort(A, dim, mode) # mode=ascend/descend

# 多项式加减<->向量加减
f = [1 2 3 4 7];
g = [0 1 4 0 1];
f + g
f - g
conv(f, g)
[Q r] = deconv(f, g)

polyder(P) # 求导
polyder(P, Q) #求P*Q的导函数
[p, q] = ployder(P, Q) #求P/Q的导函数，分子存入p,分母存入q

polyval(p, x)
polyvalm(p, x) # x是矩阵 f(matrix) = matrix^2+...

x = roots(p) #求根
p = poly(x) #根据根反求多项式

y1=interp1(x, y, x1, 'spline') #插值, spline是插值method:(linear,nearest,pchip)

p = polyfit(x, y, 3); # 拟合，返回多项式
[P, S] = polyfit(x, y, 3) # S为误差
[P, S, mu] = polyfit(x, y, 3) # mu为二元vector：[mean(X), std(X)]
```

```matlab
diff(x) # 一阶向前差分
diff(x, n) #n阶:diff(diff(x))就是2阶
diff(A, n, dim)

[l, n] = quad(f, a, b, tol, trace) # tol控制精度，trace展现积分过程，I为积分值，n为被积函数调用次数
[l, n] = quadl(f, a, b, tol, trace) 
I = integral(f, a, b) # 积分限可为inf
[I, err] = quadkg(f, a, b)
I = trapz(x, y) #梯形积分，y=f(x)但f不可知
I = integral2(f, a, b, c, d)
I = qua2d(f, a, b, c, d)
I = dblquad(f, a, b, c, d, tol)
I = integral3(f, a, b, ,c, d, e, f)
I = triplequad(f, a, b, c, d, e, f, tol)
```

注：上述积分格式：
$$
\int_a^b,\int_c^d\int_a^b,\int_e^f\int_c^d\int_a^b
$$


```matlab
Ax = b->x = A\b
[L, U] = lu(A) #上三角矩阵U和下三角矩阵L:A=LU
[L, U, P] = lu(A) # P是置换矩阵：PA=LU
Ax=b->LUx=b->x=U\(L\b)
# 迭代法略
```

```matlab
% 单变量非线性方程
x = fzeor(f, x0) % x0是迭代初始值
% 非线性方程组求解
x = fsolve(f, x0, option) % option可用optimset()
f = @(x) [f1(x), f2(x), f3(x)];

% 无约束优化：
[xmin, fmin] = fminbnd(f, x1, x2, option) # x1,x2为边界
[xmin, fmin] = fminsearch(f, x0, option) # x0是极值点初始值
[xmin, fmin] = fminunc(f, x0, option) # x0是极值点处置
[xmin, fmin] = fmincon(f, x0, A, b, Aeq, beq, Lbnd, Ubnd, NonF, option)
```

$e.g.$
$$
\min\quad f(x)=0.4x_2+x_1^2+x_2^2-x_1x_2+\frac1{30}x_1^3\\
x\in\begin{cases}
x_1+0.5x_2\ge0.4\\
0.5x_1+x_2\ge0.5\\
x_1\ge0,x_2\ge0
\end{cases}
$$

```matlab
f=@(x) 0.4*x(2)...
x0 = [0.5; 0.5];
A = [-1, -0.5; -0.5, -1];
b = [-0.4; -0.5];
lb = [0; 0];
option = optimset('Display', 'off');
[xmin, fmin] = fmincon(f, x0, A, b, [], [], lb, [], [], option)
```

```matlab
y = tpdf(x, n) # 自由度为n的t分布概率密度函数
normcdf() #正态分布累计概率密度
```

