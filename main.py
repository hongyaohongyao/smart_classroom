# This is a sample Python script.

# Press Alt+Shift+X to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
# import scipy

# from scipy.optimize import linear_sum_assignment

import tkinter


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+Shift+B to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    root = tkinter.Tk()

    Label1 = tkinter.Label(root, text='会员名称:').grid(row=0, column=0)

    v1 = tkinter.StringVar()
    p1 = tkinter.StringVar()
    e1 = tkinter.Entry(root, textvariable=v1)  # Entry 是 Tkinter 用来接收字符串等输入的控件.
    e1.grid(row=0, column=1, padx=10, pady=5)  # 设置输入框显示的位置，以及长和宽属性


    def show():
        print("会员名称:%s" % e1.get())  # 获取用户输入的信息
        root.quit()


    tkinter.Button(root, text='确认', width=10, command=show) \
        .grid(row=2, column=0, sticky='W', padx=10, pady=5)

    tkinter.Button(root, text='退出', width=10, command=root.quit) \
        .grid(row=2, column=1, sticky='E', padx=10, pady=5)

    tkinter.mainloop()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
