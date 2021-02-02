s1 = [1 2 3 4];
t1 = [2 3 1 1];
G1 = graph(s1, t1);
plot(G1);

s2 = {'学校', '电影院', '网吧', '酒店'};
t2 = {'电影院', '酒店', '酒店', 'KTV'};
G2 = graph(s2, t2);
plot(G2, 'linewidth', 2);
set(gca, 'XTick', [], 'YTick', []);

s = [1 2 3 4];
t = [2 3 1 1];
w = [3 8 9 2];
G = graph(s, t, w);
plot(G, 'EdgeLabel', G.Edges.Weight); %显示权重
set(gca, 'XTick', [], 'YTick', []);

s3 = [1 2 3 4];
t3 = [2 3 1 1];
G3 = digraph(s3, t3);
plot(G3);
