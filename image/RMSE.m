% 定义文件名和对应的标记 
files = {'001.csv', '006.csv', '688.csv', 'HS300.csv', 'sz50.csv', 'sz50eft.csv'}; 
markers = {'o', 's', 'd', '^', 'v', '>'};  % 圆形、方形、菱形、上三角、下三角、右三角 
 
% 创建图形窗口 
figure('Position', [100, 100, 1200, 600]); 
 
% 定义可能的日期格式 
dateFormats = {'yyyy-MM-dd', 'dd-MM-yyyy', 'MM/dd/yyyy', 'yyyy/MM/dd', 'yyyyMMdd'}; 
 
% 定义颜色方案 
colors = [0 0.4470 0.7410;  % 蓝色 
          0.8500 0.3250 0.0980;  % 红色 
          0.9290 0.6940 0.1250;  % 黄色 
          0.4940 0.1840 0.5560;  % 紫色 
          0.4660 0.6740 0.1880;  % 绿色 
          0.3010 0.7450 0.9330]; % 青色 
 
legendHandles = gobjects(1, length(files)); % 初始化图例句柄数组 
legendLabels = {'SSE', 'GEI', 'STAR50', 'CSI300', 'SSE50', 'SSE50ETF'}; % 更新后的图例标签
 
% 循环处理每个文件 
for i = 1:length(files) 
    % 读取CSV文件 
    data = readtable(files{i}); 
     
    % 解析日期 
    time = NaT(size(data, 1), 1); % 初始化时间数组 
    for j = 1:length(dateFormats) 
        try 
            time = datetime(data.DT, 'InputFormat', dateFormats{j}, 'Format', 'yyyy-MM-dd'); 
            break; 
        catch 
            if j == length(dateFormats) 
                error(['无法解析文件 ' files{i} ' 的日期格式']); 
            end 
        end 
    end 
     
    % 提取数据 
    RV = data.RV; 
    pRVs = data.pRVs; 
    nRVs = data.nRVs; 
     
    % 计算累积MSE 
    cumulativeMSE_PredictedRV = (cumsum((RV - pRVs).^2)); 
    cumulativeMSE_RV0 = (cumsum((RV - nRVs).^2)); 
     
    % 选择要显示标记的点（每100个点显示一个） 
    markerIndices = 1:100:length(time); 
     
    % 绘制累积MSE 
    hold on; 
    plot(time, cumulativeMSE_PredictedRV, 'Color', colors(i, :), 'LineStyle', '-'); 
    h1 = plot(time(markerIndices), cumulativeMSE_PredictedRV(markerIndices), 'Marker', markers{i}, 'LineStyle', 'none', 'Color', colors(i, :)); 
     
    plot(time, cumulativeMSE_RV0, 'Color', colors(i, :), 'LineStyle', '--'); 
    plot(time(markerIndices), cumulativeMSE_RV0(markerIndices), 'Marker', markers{i}, 'LineStyle', 'none', 'Color', colors(i, :)); 
     
    % 保存句柄用于图例 
    legendHandles(i) = h1;
end 
 
% 添加图例 
legend(legendHandles, legendLabels, 'Location', 'best'); 
 
% 设置图表属性 
xlabel('Date'); 
ylabel('Cumulative MSE'); 
grid on; 
 
% 设置x轴为时间格式和角度 
ax = gca; 
ax.XAxis.TickLabelFormat = 'yyyy-MM-dd'; 
xtickangle(45); 
 
% 自动调整图表以适应所有内容 
tight_layout();