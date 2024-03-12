from turtle import color
import matplotlib.pyplot as plt

# 设置全局字体大小为16

plt.rcParams.update({'font.size': 28})
fontsize = 28
# 给定数据
# methods = ['DIC', 'DIC-GAN', 'MSFSR', 'MSFSR-GAN', 'BSRFSR', 'BSRFSR-GAN']
# total_params = [21803.21, 21803.21, 6343.53, 6343.53, 720.22, 720.22]  # Total params / 10^3
# multi_adds = [2982.84, 2982.84, 8530.11, 8530.11, 922.64, 922.64]  # Multi-adds / 10^6
# psnr_values = [ 27.28, 25.50, 26.22, 25.33, 27.06, 26.02]  # PSNR / dB
methods = ['VapSR','RLFN','IDN','IMDN','PAN','RFDN','BSRN','LKDN','FSRNet', 'FSR-GAN', 'DIC', 'DIC-GAN', 'MSFSR', 'MSFSR-GAN', 'BSRFSR', 'BSRFSR-GAN']
total_params = [433.98,390.13,590.915,791.451,338.92,498.392,435.488,394.712,2153.93, 2153.93, 21803.21, 21803.21, 6343.53, 6343.53, 720.22, 720.22]  # Total params / 10^3
multi_adds = [109.9,95.23,1045,203.05,734.36,122.0298,106.533,100.036,15847.52, 15847.52, 2982.84, 2982.84, 8530.11, 8530.11, 922.64, 922.64]  # Multi-adds / 10^6
psnr_values = [24.72,24.39,23.76,23.93,24.63,24.41,24.64,24.62,26.31, 23.89, 27.28, 25.50, 26.22, 25.33, 27.06, 26.02]  # PSNR / dB

# 这里是关键修改的部分


# 创建散点图
fig, ax = plt.subplots(figsize=(16, 10))
max_param = max(total_params)
scale_for_5000 = 5000 / (4/5)  # 计算5000在总长度中的比例
extended_max_param = max(scale_for_5000, max_param)  # 确保所有点都能显示
plt.xlim(0, extended_max_param)  # 设置x轴范围
for i in range(len(methods)):
    if 0<multi_adds[i] <500:
        area = (10) * 9.5**2*4
        c = '#4D96FF'
        if  methods[i]== 'BSRN':
            plt.annotate(f'{methods[i]}', (total_params[i] + 1, psnr_values[i] - 0.4), fontsize=fontsize)
        else:
            plt.annotate(f'{methods[i]}', (total_params[i] + 1, psnr_values[i] + 0.1), fontsize=fontsize)
    elif 500<multi_adds[i] <5000:
        area = (30) * 9.5**2*4
        c='#FFD93D'
        if methods[i] == 'BSRFSR':

            plt.annotate(f'{methods[i]}', (total_params[i] + 1, psnr_values[i] + 0.1), fontsize=fontsize)
            plt.scatter(total_params[i], psnr_values[i], alpha=1.0, marker='*', c='r', s=1500)
        elif methods[i] == 'BSRFSR-GAN' :
            plt.annotate(f'{methods[i]}', (total_params[i] + 1, psnr_values[i] - 0.2), fontsize=fontsize)
            plt.scatter(total_params[i], psnr_values[i], alpha=1.0, marker='*', c='r', s=1500)
        elif methods[i] == 'DIC'  :
            plt.annotate(f'{methods[i]}', (total_params[i] -500, psnr_values[i] - 0.3), fontsize=fontsize)
        elif methods[i] == 'DIC-GAN':
            plt.annotate(f'{methods[i]}', (total_params[i] -2500, psnr_values[i] - 0.3), fontsize=fontsize)
        elif methods[i] == 'IDN'  :
            plt.annotate(f'{methods[i]}', (total_params[i] -900, psnr_values[i] - 0.13), fontsize=fontsize)
        elif methods[i] == 'IMDN':
            plt.annotate(f'{methods[i]}', (total_params[i] -900, psnr_values[i] +0.1), fontsize=fontsize)
        else:
            plt.annotate(f'{methods[i]}', (total_params[i] - 20, psnr_values[i] ), fontsize=fontsize)
    elif 6000<multi_adds[i] <9000:
        area = (60) * 9.5**2*4
        c='#95CD41'
        plt.annotate(f'{methods[i]}', (total_params[i] +2, psnr_values[i] + 0.2), fontsize=fontsize)
    elif multi_adds[i] >10000:
        c='#EAE7C6'
        area = (100) * 9.5**2*4
        plt.annotate(f'{methods[i]}', (total_params[i] +5, psnr_values[i] +0.15), fontsize=fontsize)

    ax.scatter(total_params[i], psnr_values[i], s=area, marker='.',c=c,alpha=0.6,label=methods[i])


h = [
    plt.plot([], [], color=c, marker='.', ms=i, alpha=a, ls='')[0] for i, c, a in zip(
        [40, 60, 80, 95], ['#4D96FF', '#FFD93D', '#95CD41', '#EAE7C6'], [0.8, 1.0, 0.6, 0.8])
]
ax.legend(
    labelspacing=0.1,
    handles=h,
    handletextpad=1,
    markerscale=1,
    fontsize=16,
    title='Multi-Adds',
    title_fontsize=fontsize,
    labels=['<500k', '500k-1000k', '1M-100M','100M+'],
    scatteryoffsets=[0.0],
    loc='lower right',
    ncol=4,
    shadow=True,
    handleheight=3.5,
    # frameon=False
    )
ax.grid( linestyle='-.', linewidth=0.5)
# 添加坐标轴标签和标题
plt.xlabel('Total Parameters (K)')
plt.ylabel('PSNR (dB)')
# plt.title('Comparison of Super-Resolution Methods')

# # 添加图例
# plt.legend()

# 显示图表

# plt.savefig('model_params', dpi=400)
plt.show()