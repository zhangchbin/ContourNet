# coding = utf-8
import tensorflow as tf
import matplotlib.pyplot as plt
# 创建GUI窗口打开图像 并显示在窗口中

from PIL import Image, ImageTk  # 导入图像处理函数库
import tkinter as tk  # 导入GUI界面函数库
import tkinter.filedialog
from control import control

# 创建窗口 设定大小并命名
window = tk.Tk()
window.title('图像显示界面')
window.geometry('600x500')
global img_png  # 定义全局变量 图像的
global filename
global gt_png
global res_png
var = tk.StringVar()  # 这时文字变量储存器
con = control()

# 创建打开图像和显示图像函数
def Open_Img():
    global img_png
    global filename
    global gt_png
    global res_png
    var.set('Have choosed!')
    Img = Image.open(filename)
    img_png = ImageTk.PhotoImage(Img)
    label_img = tk.Label(window, image=img_png)
    label_img.pack(side='left')

    name = filename.split('/')[-1].split('.')[0]
    gt_path = '/home/zhangcb/Desktop/VOCpreprocessed/PASCALContourData/groundTruth_val/' + name + '.png'
    gt = Image.open(gt_path)
    gt_png = ImageTk.PhotoImage(gt)
    #gt.show()
    label_gt = tk.Label(window, image=gt_png)
    label_gt.pack(side='left')

    con.predict_oneshot(filename) #compute result

    res_path = './result/' + name + '.png'
    res = Image.open(res_path)
    res_png = ImageTk.PhotoImage(res)
    label_res = tk.Label(window, image=res_png)
    label_res.pack(side='left')


    var.set('Compute completed!')


def Show_Img():
    global img_png
    var.set('Compute completed!')  # 设置标签的文字为 'you hit me'
    label_Img = tk.Label(window, image=img_png)
    label_Img.pack(side='left')
    label_Img2 = tk.Label(window, image=img_png)
    label_Img2.pack(side='left')


def askFile():
    global filename
    filename = tkinter.filedialog.askopenfilename()
    if filename != '':
        print(filename)
        Open_Img()


# 创建文本窗口，显示当前操作状态
Label_Show = tk.Label(window,
                      textvariable=var,  # 使用 textvariable 替换 text, 因为这个可以变化
                      bg='blue', font=('Arial', 12), width=15, height=2)
Label_Show.pack()
# 创建打开图像按钮
btn_Open = tk.Button(window,
                     text='选择图片',  # 显示在按钮上的文字
                     width=15, height=2,
                     command=askFile)  # 点击按钮式执行的命令
btn_Open.pack()  # 按钮位置
# 创建显示图像按钮
btn_Show = tk.Button(window,
                     text='计算轮廓',  # 显示在按钮上的文字
                     width=15, height=2,
                     command=Show_Img)  # 点击按钮式执行的命令
btn_Show.pack()  # 按钮位置

# 运行整体窗口
window.mainloop()
