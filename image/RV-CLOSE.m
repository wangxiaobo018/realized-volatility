% 导入数据
data = readtable('data_returns.csv');

% 将日期字符串转换为 datetime 类型
data.DT = datetime(data.DT, 'InputFormat', 'yyyy/MM/dd');

% 创建图形
figure;

% 绘制折线图
plot(data.DT, data.RV, 'b', 'LineWidth', 0.5); 

% 设置X轴和Y轴标签
xlabel('Date', 'FontName', 'Times New Roman', 'FontSize', 11);
ylabel('Returns', 'FontName', 'Times New Roman', 'FontSize', 11);

% 设置网格线
grid on;

% 获取当前坐标轴
ax = gca; 

% 设置X轴的刻度
% 计算合适的刻度间隔
numTicks = 10; % 想要显示的刻度数量
dateRange = data.DT(end) - data.DT(1);
tickInterval = ceil(length(data.DT) / numTicks);
tickIndices = 1:tickInterval:length(data.DT);
ax.XTick = data.DT(tickIndices);

% 旋转X轴刻度标签
ax.XTickLabelRotation = 45;

% 调整X轴的日期格式
ax.XAxis.TickLabelFormat = 'yyyy-MM-dd';

% 设置字体为 Times New Roman，大小为 11
ax.FontSize = 11;
ax.FontName = 'Times New Roman';

% 调整图形以适应旋转后的标签
set(gcf, 'Position', get(gcf, 'Position') .* [1 1 1 1.1]);