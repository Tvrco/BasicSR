from turtle import color
import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体大小
plt.rcParams.update({'font.size': 30})
fontsize = 30

# 给定数据
methods = ['CEBSDN','VapSR','RLFN','IDN','IMDN','PAN','RFDN','BSRN','LKDN','FSRNet', 'FSR-GAN', 'DIC', 'DIC-GAN', 'MSFSR', 'MSFSR-GAN']
total_params = [319.02, 433.98,390.13,590.915,791.451,338.92,498.392,435.488,394.712,2153.93, 2153.93, 21803.21, 21803.21, 6343.53, 6343.53]  # Total params / 10^3
multi_adds = [78.48, 109.9,95.23,1045,203.05,734.36,122.0298,106.533,100.036,15847.52, 15847.52, 2982.84, 2982.84, 8530.11, 8530.11]  # Multi-adds / 10^6
psnr_values = [26.57, 24.72,24.39,23.76,23.93,24.63,24.41,24.64,24.62,26.31, 23.89, 27.28, 25.50, 26.22, 25.33]  # PSNR / dB


fig, ax = plt.subplots(figsize=(16, 10))


def map_params(value):
    if value <= 1000:
        return value  # 0-1000不变
    elif value <= 5000:
        return 800 + (value - 1000) / (5000 - 1000) * (900 - 800)  # 1000-5000映射到800-900
    elif value < 10000:
        return 900 + (value - 5000) / (10000 - 5000) * (1000 - 900)  # 5000-10000映射到900-1000
    else:
        return 1000  # 10000+保持为1000


mapped_params = [map_params(param) for param in total_params]

for i, method in enumerate(methods):
    c = '#4D96FF' if multi_adds[i] < 100 else '#264653' if multi_adds[i] < 500 else '#FFD93D' if multi_adds[i] < 5000 else '#95CD41' if multi_adds[i] < 9000 else '#EAE7C6'
    area = (15 if multi_adds[i] < 100 else 25 if multi_adds[i] < 500 else 40 if multi_adds[i] < 5000 else 70 if multi_adds[i] < 9000 else 100) * 9.5**2 * 4
    if method == 'RFDN':
        plt.annotate(method, (mapped_params[i], psnr_values[i] - 0.3), fontsize=fontsize)
    elif method == 'RLFN':
        plt.annotate(method, (mapped_params[i]-30, psnr_values[i] - 0.3), fontsize=fontsize)
    elif method == 'BSRN':
        plt.annotate(method, (mapped_params[i]-10, psnr_values[i] - 0.3), fontsize=fontsize)
    elif method == 'VapSR':
        plt.annotate(method, (mapped_params[i]-10, psnr_values[i] + 0.2), fontsize=fontsize)
    elif method == 'LKDN':
        plt.annotate(method, (mapped_params[i]-24, psnr_values[i] + 0.2), fontsize=fontsize)
    elif method == 'DIC':
        plt.annotate(method, (mapped_params[i]-20, psnr_values[i] - 0.35), fontsize=fontsize)
    elif method == 'MSFSR':
        plt.annotate(method, (mapped_params[i]-36, psnr_values[i] + 0.2), fontsize=fontsize)
    elif method == 'DIC-GAN':
        plt.annotate(method, (mapped_params[i]-60, psnr_values[i] + 0.2), fontsize=fontsize)
    elif method == 'MSFSR-GAN':
        plt.annotate(method, (mapped_params[i]-85, psnr_values[i] +0.1), fontsize=fontsize)
    elif method == 'FSRNet':
        plt.annotate(method, (mapped_params[i]-20, psnr_values[i] + 0.4), fontsize=fontsize)
    elif method == 'FSRNet-GAN':
        plt.annotate(method, (mapped_params[i]-10, psnr_values[i] + 0.4), fontsize=fontsize)
    elif method == 'IMDN':
        plt.annotate(method, (mapped_params[i]-50, psnr_values[i] -0.3), fontsize=fontsize)
    elif method == 'CEBSDN':
        # print('CEBSDN')
        plt.scatter(mapped_params[i], psnr_values[i], alpha=1.0, marker='*', c='#C94737', s=1500)
        plt.annotate(method, (mapped_params[i]-30, psnr_values[i] + 0.2), fontsize=fontsize)
    # elif method == 'BSRFSR-GAN':
    #     plt.scatter(mapped_params[i], psnr_values[i], alpha=1.0, marker='*', c='r', s=1500)

    #     plt.annotate(method, (mapped_params[i]-30, psnr_values[i] + 0.2), fontsize=fontsize)
    elif method == 'IMDN':
        plt.annotate(method, (mapped_params[i]-25, psnr_values[i] + 0.1), fontsize=fontsize)
    else:
        plt.annotate(method, (mapped_params[i]-30, psnr_values[i] + 0.2), fontsize=fontsize)
    ax.scatter(mapped_params[i], psnr_values[i], s=area, marker='.', c=c, alpha=0.6, label=method if i == 0 else "")



ax.set_xticks([0,200,300,400, 500,600, 700,800, 900,1000])
# ax.set_xticklabels(['0', '300','400','500','1000', '2000+', ''])
ax.set_xticklabels(['0','200','300','400', '500','600', '700','2000+', '6000+','10000+'])
# ax.legend(labelspacing=0.1, handletextpad=1, markerscale=1, fontsize=fontsize, title='Multi-Adds', title_fontsize=fontsize, loc='lower right', ncol=1, shadow=True)
ax.legend(
    labelspacing=0.01,
    handles=[
    plt.plot([], [], color=c, marker='.', ms=i, alpha=a, ls='')[0] for i, c, a in zip(
        [20, 30, 40, 52,60], ['#4D96FF','#264653', '#FFD93D', '#95CD41', '#EAE7C6'], [0.8, 1.0, 0.6, 0.8, 0.3])
],
    handletextpad=1,
    markerscale=1,
    fontsize=16,
    title='Multi-Adds',
    title_fontsize=16,
    labels=['<100k', '100k-500k', '500k-1000k', '1M-100M','100M+'],
    scatteryoffsets=[0.0],
    # loc='upper left',
    ncol=1,
    shadow=True,
    handleheight=3.5,
    frameon=False
    )
ax.grid(linestyle='-.', linewidth=0.5)
plt.xlabel('Parameters (K)')
plt.ylabel('PSNR (dB)')
plt.show()
