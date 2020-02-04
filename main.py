import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams['font.family'] = ['SimSun', '宋体']
N = 0

df = pd.read_excel('../xtt/6.18整晚/6.18labview计算结果.xlsx', sheet_name=1, header=None)
pwd = '../xtt/6.18整晚/DATA -00011/BCG DATA-00011/BCGdata-00001.txt'
lines = pd.Series(np.loadtxt(pwd, dtype='float'))
plt.figure(figsize=(16, 8))
plt.title('BCG数据示例')
plt.xlabel('时间(采样点数)')
plt.ylabel('幅值(mv)')
plt.annotate(r'安静状态', xy=(75000, lines[75000]), xycoords='data', xytext=(-70, +150),
             textcoords='offset points', fontsize=36,
             arrowprops=dict(arrowstyle='->'))
plt.annotate(r'体动状态', xy=(110000, lines[110000]), xycoords='data', xytext=(-70, -150),
             textcoords='offset points', fontsize=36,
             arrowprops=dict(arrowstyle='->'))
for index, row in df.iterrows():
    if (row[3] == 1):
        plt.plot(range(0 + 45 * N, 45 + 45 * N), lines.loc[0 + 45 * N:44 + 45 * N], color='red')
    elif (row[2] == 1):
        plt.plot(range(0 + 45 * N, 45 + 45 * N), lines.loc[0 + 45 * N:44 + 45 * N], color='yellow')
    else:
        plt.plot(range(0 + 45 * N, 45 + 45 * N), lines.loc[0 + 45 * N:44 + 45 * N], color='blue')
    N = N + 1
plt.show()
