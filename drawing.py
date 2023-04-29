# module train
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("D:\\exp\\results.csv")
data = np.array(data)
x = data[:,0]
y = data[:,1]
plt.figure(figsize=(10,5),dpi=300)
bwidth = 0.5
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwidth)
ax.spines['left'].set_linewidth(bwidth)
ax.spines['top'].set_linewidth(bwidth)
ax.spines['right'].set_linewidth(bwidth)
plt.plot(x,y,label='训练集损失值',color='red')
plt.xticks(x[::5],fontsize=6)
plt.yticks(y[::25],fontsize=6)
plt.xlabel('训练次数',fontsize=5)
plt.ylabel('损失值',fontsize=5,loc="center",rotation=90)
plt.title('训练集损失随训练次数变化图',fontsize=8)
plt.grid(True,linestyle='-',color='black',alpha=0.1)
plt.legend(loc="best",fontsize=6,frameon=False)
#坐标轴中文显示
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#调整图片大小
plt.gcf().subplots_adjust(left=0.1,top=0.90,bottom=0.2, right=0.8)
plt.show()