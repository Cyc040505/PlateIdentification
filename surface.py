import cv2
import time
import algorithm
import tkinter as tk
from tkinter.filedialog import *
from tkinter import ttk, LEFT, BOTH, TOP, RIGHT
from PIL import Image, ImageTk
import threading


# 创建GUI界面
class Surface(ttk.Frame):
    pic_path = ""
    view_high = 600
    view_wide = 600
    update_time = 0
    thread = None
    thread_run = False
    camera = None
    color_transform = {
        "green": ("绿牌", "#55FF55"),
        "yellow": ("黄牌", "#FFFF00"),
        "blue": ("蓝牌", "#6666FF"),
        "hk_yellow": ("黄牌(港澳地区)", "#FFFF00")}

    def __init__(self, win):
        # 设置窗口和布局
        ttk.Frame.__init__(self, win)
        frame_left = ttk.Frame(self)
        frame_right1 = ttk.Frame(self)
        frame_right2 = ttk.Frame(self)
        win.title("车牌识别")
        win.geometry("1200x800")
        self.pack(fill=tk.BOTH, expand=tk.YES, padx="5", pady="5")
        frame_left.pack(side=LEFT, expand=1, fill=BOTH)
        frame_right1.pack(side=TOP, expand=1, fill=tk.Y)
        frame_right2.pack(side=RIGHT, expand=0)

        self.original_image_frame = ttk.Frame(frame_left, borderwidth=2, relief="solid")
        self.original_image_frame.pack(fill=tk.BOTH, expand=tk.YES, padx=5, pady=5)
        ttk.Label(self.original_image_frame, text='原图：').pack(anchor="nw")
        ttk.Label(frame_right1, text='车牌位置：').grid(column=0, row=0, sticky=tk.W)

        # 创建按钮和标签
        from_pic_ctl = ttk.Button(frame_right2, text="识别大陆车牌", width=20, command=self.from_pic)
        self.image_ctl = ttk.Label(self.original_image_frame)
        self.image_ctl.pack(anchor="nw", fill=tk.BOTH, expand=tk.YES)
        self.roi_ctl = ttk.Label(frame_right1)
        self.roi_ctl.grid(column=0, row=1, sticky=tk.W)
        ttk.Label(frame_right1, text='识别结果：').grid(column=0, row=2, sticky=tk.W)
        self.r_ctl = ttk.Label(frame_right1, text="")
        self.r_ctl.grid(column=0, row=3, sticky=tk.W)
        self.color_ctl = ttk.Label(frame_right1, text="", width=20)
        self.color_ctl.grid(column=0, row=4, sticky=tk.W)
        from_pic_ctl.pack(anchor="se", pady="5")
        # 添加港澳车牌识别按钮
        self.hk_plate_ctl = ttk.Button(frame_right2, text="识别港澳车牌", width=20, command=self.from_hk_pic)
        self.hk_plate_ctl.pack(anchor="se", pady="5")

        self.predictor = algorithm.CardPredictor()

    # 获取图像，转换为Tkinter可用格式
    def get_imgtk(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)
        wide = imgtk.width()
        high = imgtk.height()
        if wide > self.view_wide or high > self.view_high:
            wide_factor = self.view_wide / wide
            high_factor = self.view_high / high
            factor = min(wide_factor, high_factor)

            wide = int(wide * factor)
            if wide <= 0:
                wide = 1
            high = int(high * factor)
            if high <= 0:
                high = 1
            im = im.resize((wide, high), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=im)
        return imgtk

    # 显示车牌识别结果
    def show_roi(self, r, roi, color):
        if r:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = Image.fromarray(roi)
            self.imgtk_roi = ImageTk.PhotoImage(image=roi)
            self.roi_ctl.configure(image=self.imgtk_roi, state='enable')
            self.r_ctl.configure(text=str(r))
            self.update_time = time.time()
            try:
                c = self.color_transform[color]
                self.color_ctl.configure(text=c[0], background=c[1], state='enable')
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                self.color_ctl.configure(state='disabled')
        elif self.update_time + 8 < time.time():
            self.roi_ctl.configure(state='disabled')
            self.r_ctl.configure(text="")
            self.color_ctl.configure(state='disabled')
        print("识别完成")

    # 从图片文件读取图像并识别
    def from_pic(self):
        default_folder = "dataset/inland"
        self.pic_path = askopenfilename(title="选择识别图片", filetypes=[("jpg图片", "*.jpg")],
                                        initialdir=default_folder)
        if self.pic_path:
            print("开始识别")
            self.reset_ui()
            self.thread_run = True
            self.thread = threading.Thread(target=self.process_image, args=(self.pic_path,))
            self.thread.start()

    # 读取港澳车牌并识别
    def from_hk_pic(self):
        default_folder = "dataset/hk"
        self.pic_path = askopenfilename(title="选择识别图片", filetypes=[("jpg图片", "*.jpg")],
                                        initialdir=default_folder)
        if self.pic_path:
            print("开始识别")
            self.reset_ui()
            self.thread_run = True
            self.thread = threading.Thread(target=self.process_hk_image, args=(self.pic_path,))
            self.thread.start()

    # 重置UI组件
    def reset_ui(self):
        self.image_ctl.configure(image='')  # 清空原图显示
        self.roi_ctl.configure(state='disabled')  # 禁用车牌位置显示
        self.r_ctl.configure(text="")  # 清空识别结果
        self.color_ctl.configure(state='disabled')  # 禁用颜色显示

    # 处理图像识别任务
    def process_image(self, pic_path):
        img_bgr = algorithm.imreadex(pic_path)
        self.imgtk = self.get_imgtk(img_bgr)
        self.image_ctl.configure(image=self.imgtk)
        resize_rates = (1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4)
        for resize_rate in resize_rates:
            print("resize_rate:", resize_rate)
            try:
                r, roi, color = self.predictor.predict(img_bgr, resize_rate)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                continue
            if r:
                break
        r, roi, color = self.predictor.predict(img_bgr, 1)
        self.show_roi(r, roi, color)
        self.thread_run = False

    # 处理港澳车牌识别任务
    def process_hk_image(self, pic_path):
        img_bgr = algorithm.imreadex(pic_path)
        self.imgtk = self.get_imgtk(img_bgr)
        self.image_ctl.configure(image=self.imgtk)
        r, roi, color = self.predictor.predict_hk(img_bgr, 1)
        self.show_roi(r, roi, color)
        self.thread_run = False


# 关闭窗口时的处理
def close_window():
    print("车牌识别系统关闭")
    if surface.thread_run:
        surface.thread_run = False
        surface.thread.join(2.0)
    win.destroy()


if __name__ == '__main__':
    print("车牌识别系统启动")
    win = tk.Tk()
    surface = Surface(win)
    win.protocol('WM_DELETE_WINDOW', close_window)
    win.mainloop()
