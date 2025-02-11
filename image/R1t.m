data = readtable('rv_12.csv');
% ��ȡ��Ҫ�ı���
R1t = data.R1t;
R2t = data.R2t;
RV = data.RV;

% ������ƾ���
X = [ones(size(R1t)), R1t];  % ����ƽ����

% �������Իع�
beta = regress(RV, X);

% ����������
[sortedR1t, sortIndex] = sort(R1t);
sortedRV = RV(sortIndex);
sortedR2t = R2t(sortIndex);
sortedX = X(sortIndex, :);

% ������ϵ�����ֵ
sortedFittedRV = sortedX * beta;

% ����ͼ��
figure;

% ����RV�������
plot(sortedR1t, sortedFittedRV, 'r', 'LineWidth', 2); % �����������
hold on;

% �����ɫ�����R2tɢ��ͼ
scatter(sortedR1t, sortedRV, 20, sortedR2t, 'filled'); % ʹ��R2t��ֵ������ɫӳ��
colormap(jet); % ʹ��jet��ɫͼ
cb = colorbar; % ��ʾ��ɫ��
cb.Label.String = '$R_{2,t}$'; % ʹ��LaTeX��ʽ
cb.Label.Interpreter = 'latex';
cb.FontSize = 11; % ������ɫ�������С
cb.FontName = 'Times New Roman'; % ������ɫ������Ϊ Times New Roman

% ������ɫ���Ŀ�ѧ�������̶�
baseExponent = floor(log10(max(abs(sortedR2t))));
baseMultiplier = 10^baseExponent;
cb.Ticks = linspace(min(sortedR2t), max(sortedR2t), 5); % ����5���̶�
cb.TickLabels = arrayfun(@(x) sprintf('%.1f', x / baseMultiplier), cb.Ticks, 'UniformOutput', false); % ��ʽ���̶ȱ�ǩ
title(cb, ['\times10^{' num2str(baseExponent) '}']); % ���ָ����ǩ

% ���ͼ�������Ͻ�
legend({'RV Fitted Curve', '$R_{2,t}$ Data Points'}, ...
    'Location', 'Northeast', 'Interpreter', 'latex', 'FontSize', 9, 'FontName', 'Times New Roman');

% ��������ΪTimes New Roman�������������СΪ11
ax = gca; % ��ȡ��ǰ������
ax.FontName = 'Times New Roman';
ax.FontSize = 11; % ����������̶������С
ax.XLabel.FontName = 'Times New Roman';
ax.YLabel.FontName = 'Times New Roman';
ax.XLabel.FontSize = 11; % X���ǩ�����С
ax.YLabel.FontSize = 11; % Y���ǩ�����С

% ͼ������
xlabel('$R_{1,t}$', 'Interpreter', 'latex', 'FontSize', 11);
ylabel('RV', 'Interpreter', 'latex', 'FontSize', 11);
grid on;
hold off;
