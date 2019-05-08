from matplotlib import pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif']=['SimHei']  #plt绘制图像时title显示中文标题,解决显示异常bug
plt.rcParams['axes.unicode_minus'] = False

'''
简单函数图像绘制
'''
# x=np.linspace(0,20,100)
# y1=x**2/20+np.sin(x)
# y2=x**2/20-np.sin(x)
# plt.figure(figsize=(9,3),dpi=200)
# plt.subplot(1,3,1)  #多表绘制,1行3列,第1列
# plt.plot(x,y1,'r',label='y1=x**2/20+sin(x)')
# plt.plot(x,y2,'g--',label='y2=x**2/20-sin(x)')
# plt.xlabel(xlabel='x')
# plt.ylabel(ylabel='y')
# plt.title('y1  y2')
# plt.legend(loc='upper left')
# plt.subplot(1,3,3)  #多表绘制,1行3列,第2列
# plt.plot(x,y1,'r',label='y1=x**2/20+sin(x)')
# plt.xlabel(xlabel='x')
# plt.ylabel(ylabel='y')
# plt.title('y1')
# plt.legend(loc='upper left')
# plt.subplot(1,3,2)  #多表绘制,1行3列,第3列
# plt.plot(x,y2,'g--',label='y2=x**2/20-sin(x)')
# plt.xlabel(xlabel='x')
# plt.ylabel(ylabel='y')
# plt.title('y2')
# plt.legend(loc='upper left')
# plt.plot(x,y2,'g--',label='y2=x**2/20-sin(x)')
# plt.tight_layout()
# plt.savefig('简单函数绘制图.jpg')
# plt.show()

'''
轴的刻度设置成对数刻度，调用 set_xscale 与 set_yscale 设置刻度，参数选择 “log”
'''
# x=np.linspace(0,5,50)
# y=(x+2)**2/10+np.sin(x*np.pi)
# fig, axes = plt.subplots(1, 2, figsize=(6,3),dpi=200)
# axes[0].plot(x, y,label='y=x^2/10+sin(x*pi)')
# axes[0].plot( x, np.exp(x),label='y=e^x')
# axes[0].legend(loc=2)
# axes[0].set_title("Normal scale")
# axes[1].plot(x, y, x, np.exp(x))
# axes[1].set_yscale('log')  #y坐标轴对数形式展示
# axes[1].set_title("Logarithmic scale (y)")
# plt.savefig('坐标轴对数形式展示.jpg')
# plt.tight_layout()
# plt.show()

'''
自定义坐标轴标签符号
set_xticks set_yticks方法显示坐标符号位置
set_xticklabels set_yticklabels为每个坐标设置符号
'''
# fig, ax=plt.subplots(figsize=(6,3),dpi=200)
# x=np.linspace(0,5,100)
# ax.plot(x,x**2,x,x**3)
# ax.set_xticks([1,2,3,4,5])
# ax.set_xticklabels([r"$\alpha$",r"$\beta$",r"$\gamma$",r"$\delta$",r"$\epsilon$"],fontsize=18)
# yticks=[0,50,100,150]
# ax.set_yticks(yticks)
# ax.set_yticklabels(["$%.1f$"%y for y in yticks],fontsize=18)
# ax.set_title('自定义坐标轴')
# plt.tight_layout()
# plt.savefig('自定义坐标轴.jpg')
# plt.show()

'''
双坐标轴显示
twinx 与 twiny 函数能设置双坐标轴：
'''
#
# fig,ax=plt.subplots(figsize=(6,3),dpi=200)
# x=np.linspace(-5,5,100)
# ax.plot(x,x**3,lw=2,color='g')
# ax.set_ylabel('area$(m^2)$',color='g',fontsize=18)
# for label in ax.get_yticklabels():
#     label.set_color('g')
# ax1=ax.twinx()
# ax1.plot(x,np.tan(x*np.pi),lw=2,color='r')
# ax1.set_ylabel('$tan(x)$',color='r',fontsize=18)
# for label in ax1.get_yticklabels():
#     label.set_color('r')
# plt.tight_layout()
# plt.savefig('双坐标轴显示.jpg')
# plt.show()

'''
坐标轴居中显示
spines,set_ticks_pisition
'''
# fig, ax = plt.subplots(figsize=(6,3),dpi=200)
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.spines['bottom'].set_position(('data',0)) # set position of x spine to x=0
# ax.yaxis.set_ticks_position('left')
# ax.spines['left'].set_position(('data',0))   # set position of y spine to y=0
# x = np.linspace(-5, 5, 24)
# ax.plot(x, x**3+50*np.sin(x*np.pi*0.5), color="purple", lw=1, ls='--', marker='s', markersize=4,
#         markerfacecolor="w", markeredgewidth=1, markeredgecolor="blue")
# plt.tight_layout()
# plt.savefig('坐标轴居中显示.jpg')
# plt.show()

'''
plt.scatter() 散点图
plt.bar(n,f(n),align,width,color,alpha)柱状图
plt.step(n,f(n),ls,lw,color)单步界跃图
plt.fill_between(x,y1,y2,color...)图形函数区间着色
plt.hist()制定每个箱子(bin)分布数据,对应X轴
'''
# x=np.linspace(-5,5,50)
# n=np.arange(0,6,1)
# print(n)
# fig, axes = plt.subplots(2, 3, figsize=(9,6))
# axes[0,0].scatter(x, x + 0.25*np.random.randn(len(x)),marker='+',color='r',lw=1,alpha=0.8)
# axes[0,0].set_title("scatter")
# axes[0,1].step(n, n**2, lw=2)
# axes[0,1].set_title("step")
# axes[1,0].bar(n, n**2+np.sin((n+1)*np.pi/2)*(5-n), align="center", width=0.5, alpha=0.5)
# axes[1,0].set_title("bar")
# axes[1,1].fill_between(x, x**2, np.sin(x*np.pi), color="green", alpha=0.5)
# axes[1,1].set_title("fill_between")
# m=np.random.randn(10000)+5
# axes[0,2].hist(m,bins=50,color='r',alpha=0.6)
# axes[0,2].set_title('Default histogram')
# axes[1,2].hist(m,bins=50,cumulative=True)
# axes[1,2].set_title('Cumulative detailed histogram')
# plt.tight_layout()
# plt.savefig('柱状图&点阵图等图示.jpg')
# plt.show()

'''
饼状图
'''
plt.figure(figsize=(6,6),dpi=200)
label=['Beijing','Shanghai',"Suzhou","Wuhan","Hefei"]
percent=[43,20,12,16,9]
colors=['yellowgreen','gold','lightskyblue','lightcoral','red']
explode=[0,0.1,0,0,0]
plt.pie(x=percent,labels=label,colors=colors,explode=explode,autopct='%.1f%%',shadow=True,startangle=0)
plt.axis('equal')
plt.tight_layout()
plt.savefig('饼状图.jpg')
plt.show()