# coding = utf-8
import tensorflow as tf
import matplotlib.pyplot as plt
# create GUI

from PIL import Image, ImageTk 
import tkinter as tk  
import tkinter.filedialog
from control import control

# set the window, size, name
window = tk.Tk()
window.title('image show window')
window.geometry('600x500')
global img_png  # denote the image
global filename
global gt_png
global res_png
var = tk.StringVar()  # text 
con = control()

# open the image and show it
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
    var.set('Compute completed!')  # set the name as 'you hit me'
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


# create text window
Label_Show = tk.Label(window,
                      textvariable=var,  
                      bg='blue', font=('Arial', 12), width=15, height=2)
Label_Show.pack()

btn_Open = tk.Button(window,
                     text='select the image',  
                     width=15, height=2,
                     command=askFile)  
btn_Open.pack()  

btn_Show = tk.Button(window,
                     text='run',  
                     width=15, height=2,
                     command=Show_Img) 
btn_Show.pack() 

window.mainloop()
