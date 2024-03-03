import numpy as np
import matplotlib.pyplot as plt

# 0 设置字体
plt.rc('font', family='Times New Roman', size=15)

# 1.1 定义sigmoid函数
def sigmoid(x):
    return 1. / (1 + np.exp(-x))

# 1.2 定义tanh函数
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# 1.3 定义relu函数
def relu(x):
    return np.where(x < 0, 0, x)

# 1.4 定义prelu函数
def prelu(x):
    return np.where(x < 0, x * 0.5, x)

# 1.5 定义gelu函数
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
# 2 绘制所有激活函数
def plot_functions():
    x = np.arange(-10, 10, 0.1)
    functions = [sigmoid, tanh, relu, prelu, gelu]
    titles = ['Sigmoid', 'Tanh', 'ReLU', 'PReLU', 'GELU']

    fig = plt.figure(figsize=(20, 4))  # 设置画布大小，确保足够宽以容纳所有子图

    for i, func in enumerate(functions, 1):
        ax = fig.add_subplot(1, 5, i)  # 1行5列
        y = func(x)
        ax.plot(x, y, color="black", lw=3)
        # ax.set_title(titles[i-1])

        # 设置坐标轴
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        if i in [1, 2]:  # Sigmoid和Tanh特殊处理
            ax.spines['left'].set_position(('data', 0))
            ax.spines['bottom'].set_position(('data', 0))
        else:
            ax.spines['left'].set_position(('data', 0))

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim([-10.05, 10.05])
        plt.ylim(min(y) - 0.1, max(y) + 0.1)
        plt.tight_layout()
    plt.savefig('C:/Users/87306/OneDrive/毕业论文/第二大点/activation_functions.png', dpi=400)
    plt.show()

# 3 运行程序
plot_functions()