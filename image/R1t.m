data = readtable('rv_12.csv');
% 提取需要的变量
R1t = data.R1t;
R2t = data.R2t;
RV = data.RV;

% 构建设计矩阵
X = [ones(size(R1t)), R1t];  % 不加平方项

% 进行线性回归
beta = regress(RV, X);

% 对数据排序
[sortedR1t, sortIndex] = sort(R1t);
sortedRV = RV(sortIndex);
sortedR2t = R2t(sortIndex);
sortedX = X(sortIndex, :);

% 计算拟合的曲线值
sortedFittedRV = sortedX * beta;

% 创建图形
figure;

% 绘制RV拟合曲线
plot(sortedR1t, sortedFittedRV, 'r', 'LineWidth', 2); % 绘制拟合曲线
hold on;

% 添加颜色渐变的R2t散点图
scatter(sortedR1t, sortedRV, 20, sortedR2t, 'filled'); % 使用R2t的值进行颜色映射
colormap(jet); % 使用jet颜色图
cb = colorbar; % 显示颜色条
cb.Label.String = '$R_{2,t}$'; % 使用LaTeX格式
cb.Label.Interpreter = 'latex';
cb.FontSize = 11; % 设置颜色条字体大小
cb.FontName = 'Times New Roman'; % 设置颜色条字体为 Times New Roman

% 设置颜色条的科学记数法刻度
baseExponent = floor(log10(max(abs(sortedR2t))));
baseMultiplier = 10^baseExponent;
cb.Ticks = linspace(min(sortedR2t), max(sortedR2t), 5); % 设置5个刻度
cb.TickLabels = arrayfun(@(x) sprintf('%.1f', x / baseMultiplier), cb.Ticks, 'UniformOutput', false); % 格式化刻度标签
title(cb, ['\times10^{' num2str(baseExponent) '}']); % 添加指数标签

% 添加图例到右上角
legend({'RV Fitted Curve', '$R_{2,t}$ Data Points'}, ...
    'Location', 'Northeast', 'Interpreter', 'latex', 'FontSize', 9, 'FontName', 'Times New Roman');

% 设置字体为Times New Roman，坐标轴字体大小为11
ax = gca; % 获取当前坐标轴
ax.FontName = 'Times New Roman';
ax.FontSize = 11; % 设置坐标轴刻度字体大小
ax.XLabel.FontName = 'Times New Roman';
ax.YLabel.FontName = 'Times New Roman';
ax.XLabel.FontSize = 11; % X轴标签字体大小
ax.YLabel.FontSize = 11; % Y轴标签字体大小

% 图形美化
xlabel('$R_{1,t}$', 'Interpreter', 'latex', 'FontSize', 11);
ylabel('RV', 'Interpreter', 'latex', 'FontSize', 11);
grid on;
hold off;
