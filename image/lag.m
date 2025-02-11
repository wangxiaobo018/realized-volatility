% ��������
data = readtable('data_returns.csv');

% ����ACF������ͺ�Ϊ500
[acf, lags] = autocorr(data.RV, 500);

% ����ͼ��
figure;

% ����ACFͼ
stem(lags, acf, 'b', 'LineWidth', 0.5, 'MarkerSize', 2); % ʹ����ɫ

% ����������
grid on;

% ����X���Y���ǩ
xlabel('Lag', 'FontName', 'Times New Roman', 'FontSize', 11);
ylabel('ACF', 'FontName', 'Times New Roman', 'FontSize', 11);

% ��������Ϊ Times New Roman����СΪ 11
ax = gca; 
ax.FontSize = 11;
ax.FontName = 'Times New Roman';

% ���ˮƽ�߱�ʾ�������䣨95%����ˮƽ��
conf = 1.96/sqrt(length(data.RV));
hold on;
plot([0 500], [conf conf], 'r--', 'LineWidth', 0.5);
plot([0 500], [-conf -conf], 'r--', 'LineWidth', 0.5);
hold off;