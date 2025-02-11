% �����ļ����Ͷ�Ӧ�ı�� 
files = {'001.csv', '006.csv', '688.csv', 'HS300.csv', 'sz50.csv', 'sz50eft.csv'}; 
markers = {'o', 's', 'd', '^', 'v', '>'};  % Բ�Ρ����Ρ����Ρ������ǡ������ǡ������� 
 
% ����ͼ�δ��� 
figure('Position', [100, 100, 1200, 600]); 
 
% ������ܵ����ڸ�ʽ 
dateFormats = {'yyyy-MM-dd', 'dd-MM-yyyy', 'MM/dd/yyyy', 'yyyy/MM/dd', 'yyyyMMdd'}; 
 
% ������ɫ���� 
colors = [0 0.4470 0.7410;  % ��ɫ 
          0.8500 0.3250 0.0980;  % ��ɫ 
          0.9290 0.6940 0.1250;  % ��ɫ 
          0.4940 0.1840 0.5560;  % ��ɫ 
          0.4660 0.6740 0.1880;  % ��ɫ 
          0.3010 0.7450 0.9330]; % ��ɫ 
 
legendHandles = gobjects(1, length(files)); % ��ʼ��ͼ��������� 
legendLabels = {'SSE', 'GEI', 'STAR50', 'CSI300', 'SSE50', 'SSE50ETF'}; % ���º��ͼ����ǩ
 
% ѭ������ÿ���ļ� 
for i = 1:length(files) 
    % ��ȡCSV�ļ� 
    data = readtable(files{i}); 
     
    % �������� 
    time = NaT(size(data, 1), 1); % ��ʼ��ʱ������ 
    for j = 1:length(dateFormats) 
        try 
            time = datetime(data.DT, 'InputFormat', dateFormats{j}, 'Format', 'yyyy-MM-dd'); 
            break; 
        catch 
            if j == length(dateFormats) 
                error(['�޷������ļ� ' files{i} ' �����ڸ�ʽ']); 
            end 
        end 
    end 
     
    % ��ȡ���� 
    RV = data.RV; 
    pRVs = data.pRVs; 
    nRVs = data.nRVs; 
     
    % �����ۻ�MSE 
    cumulativeMSE_PredictedRV = (cumsum((RV - pRVs).^2)); 
    cumulativeMSE_RV0 = (cumsum((RV - nRVs).^2)); 
     
    % ѡ��Ҫ��ʾ��ǵĵ㣨ÿ100������ʾһ���� 
    markerIndices = 1:100:length(time); 
     
    % �����ۻ�MSE 
    hold on; 
    plot(time, cumulativeMSE_PredictedRV, 'Color', colors(i, :), 'LineStyle', '-'); 
    h1 = plot(time(markerIndices), cumulativeMSE_PredictedRV(markerIndices), 'Marker', markers{i}, 'LineStyle', 'none', 'Color', colors(i, :)); 
     
    plot(time, cumulativeMSE_RV0, 'Color', colors(i, :), 'LineStyle', '--'); 
    plot(time(markerIndices), cumulativeMSE_RV0(markerIndices), 'Marker', markers{i}, 'LineStyle', 'none', 'Color', colors(i, :)); 
     
    % ����������ͼ�� 
    legendHandles(i) = h1;
end 
 
% ���ͼ�� 
legend(legendHandles, legendLabels, 'Location', 'best'); 
 
% ����ͼ������ 
xlabel('Date'); 
ylabel('Cumulative MSE'); 
grid on; 
 
% ����x��Ϊʱ���ʽ�ͽǶ� 
ax = gca; 
ax.XAxis.TickLabelFormat = 'yyyy-MM-dd'; 
xtickangle(45); 
 
% �Զ�����ͼ������Ӧ�������� 
tight_layout();