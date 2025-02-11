% 导入数据
data = readtable('data_returns.csv');

% 计算ACF，最大滞后为500
[acf, lags] = autocorr(data.RV, 500);

% 创建图形
figure;

% 绘制ACF图
stem(lags, acf, 'b', 'LineWidth', 0.5, 'MarkerSize', 2); % 使用蓝色

% 设置网格线
grid on;

% 设置X轴和Y轴标签
xlabel('Lag', 'FontName', 'Times New Roman', 'FontSize', 11);
ylabel('ACF', 'FontName', 'Times New Roman', 'FontSize', 11);

% 设置字体为 Times New Roman，大小为 11
ax = gca; 
ax.FontSize = 11;
ax.FontName = 'Times New Roman';

% 添加水平线表示置信区间（95%置信水平）
conf = 1.96/sqrt(length(data.RV));
hold on;
plot([0 500], [conf conf], 'r--', 'LineWidth', 0.5);
plot([0 500], [-conf -conf], 'r--', 'LineWidth', 0.5);
hold off;