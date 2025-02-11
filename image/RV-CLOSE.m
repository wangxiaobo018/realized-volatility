% ��������
data = readtable('data_returns.csv');

% �������ַ���ת��Ϊ datetime ����
data.DT = datetime(data.DT, 'InputFormat', 'yyyy/MM/dd');

% ����ͼ��
figure;

% ��������ͼ
plot(data.DT, data.RV, 'b', 'LineWidth', 0.5); 

% ����X���Y���ǩ
xlabel('Date', 'FontName', 'Times New Roman', 'FontSize', 11);
ylabel('Returns', 'FontName', 'Times New Roman', 'FontSize', 11);

% ����������
grid on;

% ��ȡ��ǰ������
ax = gca; 

% ����X��Ŀ̶�
% ������ʵĿ̶ȼ��
numTicks = 10; % ��Ҫ��ʾ�Ŀ̶�����
dateRange = data.DT(end) - data.DT(1);
tickInterval = ceil(length(data.DT) / numTicks);
tickIndices = 1:tickInterval:length(data.DT);
ax.XTick = data.DT(tickIndices);

% ��תX��̶ȱ�ǩ
ax.XTickLabelRotation = 45;

% ����X������ڸ�ʽ
ax.XAxis.TickLabelFormat = 'yyyy-MM-dd';

% ��������Ϊ Times New Roman����СΪ 11
ax.FontSize = 11;
ax.FontName = 'Times New Roman';

% ����ͼ������Ӧ��ת��ı�ǩ
set(gcf, 'Position', get(gcf, 'Position') .* [1 1 1 1.1]);