import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

'''
对数据的demo展示
'''
matplotlib.rcParams['font.family'] = ['SimSun', '宋体']
N = 0

# df = pd.read_excel('../xtt/6.18整晚/6.18labview计算结果.xlsx', sheet_name=1, header=None)

# lines = pd.Series(np.loadtxt(pwd, dtype='float'))
'''
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
'''

'''
导入所有已标记数据
'''
sheets = [i for i in range(1, 41) if i != 2 and i != 3]
names = locals()
data_all = pd.DataFrame()


# 峰值
def peak(x):
    return np.max(x) - np.min(x)


# 整流平均值
def arv(x):
    return np.mean(np.abs(x))


# 均方根值
def rms(x):
    return np.sqrt(np.mean(x ** 2))


# 过零点数
def crosszeros(x):
    N = 0
    for i in range(x.size - 1):
        if x.iloc[i] * x.iloc[i + 1] <= 0:
            N += 1
    return N


for i in sheets:
    start = time.time()
    names['raw' + str(i)] = pd.read_excel('../xtt/6.18整晚/6.18labview计算结果.xlsx', sheet_name=i, header=None)
    if i < 10:
        pwd = '../xtt/6.18整晚/DATA -00011/BCG DATA-00011/BCGdata-0000{}.txt'.format(i)
    else:
        pwd = '../xtt/6.18整晚/DATA -00011/BCG DATA-00011/BCGdata-000{}.txt'.format(i)
    lines = pd.Series(np.loadtxt(pwd, dtype='float'))
    series = pd.Series([i // 45 for i in range(names.get('raw' + str(i)).index.size * 45)])
    lines.drop(index=list(range(series.size, lines.size)), inplace=True)
    df = pd.DataFrame()
    df['raw'] = lines
    df['marked'] = series

    # 最大值、最小值、平均值、方差、标准差、平均绝对偏差、峰度、偏度、峰值、整流平均值、均方根值、波形因子、峰值因子、脉冲因子、过零点数
    # 一阶差分的最大值、最小值、平均值、方差、标准差、平均绝对偏差、峰度、偏度
    # 快速傅里叶变换后的方差、平均值、最大值、最小值
    agg = df.groupby(['marked'])['raw'].agg(
        [
            pd.DataFrame.max, pd.DataFrame.min, pd.DataFrame.mean, pd.DataFrame.var, pd.DataFrame.std, pd.DataFrame.mad,
            pd.DataFrame.kurtosis, pd.DataFrame.skew, peak, arv, rms,
            lambda x: rms(x) / arv(x),
            lambda x: peak(x) / rms(x),
            lambda x: peak(x) / arv(x),
            crosszeros,
            lambda x: x.diff().max(), lambda x: x.diff().min(), lambda x: x.diff().mean(), lambda x: x.diff().var(),
            lambda x: x.diff().std(), lambda x: x.diff().mad(), lambda x: x.diff().kurtosis(),
            lambda x: x.diff().skew(),
            lambda x: abs(np.var(np.fft.fft(x))),
            lambda x: abs(np.mean(np.fft.fft(x))),
            lambda x: abs(np.max(np.fft.fft(x))),
            lambda x: abs(np.min(np.fft.fft(x)))
        ]).rename(columns={"<lambda_0>": "form_factor", "<lambda_1>": "crest_factor", "<lambda_2>": "maichong_factor",
                           "<lambda_3>": "diff_max", "<lambda_4>": "diff_min", "<lambda_5>": "diff_mean",
                           "<lambda_6>": "diff_var", "<lambda_7>": "diff_std", "<lambda_8>": "diff_mad",
                           "<lambda_9>": "diff_kurt", "<lambda_10>": "diff_skew", "<lambda_11>": "fft_var",
                           "<lambda_12>": "fft_mean", "<lambda_13>": "fft_max", "<lambda_14>": "fft_min"
                           })
    names['data' + str(i)] = pd.concat([names.get('raw' + str(i)).iloc[:, [2, 3]], agg], axis=1).rename(
        columns={2: "is_leave_mattress", 3: "is_body_move"})
    print('第{}个表的数据处理完毕，用时{}s'.format(i, time.time() - start))
    data_all = data_all.append(names.get('data' + str(i)), ignore_index=True)

data_all.to_csv('data_all.csv')
