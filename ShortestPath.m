function [P, D, d] = find_shortest_path(G, s, t)
    %P是路径，d是距离
    [P, d] = shortestpath(G, s, t);
    myplot = plot(G, 'EdgeLabel', G.Edges.Weight, 'linewidth', 2); %赋予图变量
    highlight(myplot, P, 'EdgeColor', 'r'); %对路径进行高亮处理
    D = distance(G); %距离矩阵    
end
