import matplotlib
import json
matplotlib.use('TkAgg')
# from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# Implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler


from matplotlib.figure import Figure

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk
    from tkinter import ttk

import numpy as np
import cv2


def MinMaxProjectorCalibration(patterns, cameras, projector):
    end = False
    pattern_num = 0

    def quit(root):
        nonlocal end
        end = True
        root.quit()     # Stops mainloop
        root.destroy()  # This is necessary on Windows to prevent
                        # Fatal Python Error: PyEval_RestoreThread: NULL tstate

    root = Tk.Tk()
    root.wm_title("Embedding in TK")

    # Callback function to show next pattern
    def next_pattern(root):
        nonlocal pattern_num
        pattern_num = pattern_num + 1
        if pattern_num == len(patterns):
            pattern_num = 0

    f = Figure(figsize=(5, 4), dpi=100)

    # A tk.DrawingArea
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

    button = Tk.Button(master=root, text='Quit', command=lambda: quit(root))
    button.pack(side=Tk.BOTTOM)
    button = Tk.Button(master=root, text='Next', command=lambda: next_pattern(root))
    button.pack(side=Tk.BOTTOM)

    def slider_max_changed(event):
        projector.max_image_brightness = current_max_brightness.get()

    def slider_min_changed(event):
        projector.min_image_brightness = current_min_brightness.get()

    current_max_brightness = Tk.DoubleVar(value=projector.max_image_brightness)
    current_min_brightness = Tk.DoubleVar(value=projector.min_image_brightness)
    scale_max = Tk.Scale(root, orient="horizontal",
                            from_=0, to=1.0,
                            digits=4,
                            resolution=0.01,
                            variable=current_max_brightness,
                            command=slider_max_changed,
                            length=300)
    scale_min = Tk.Scale(root, orient="horizontal",
                            from_=0, to=1.0, 
                            digits=4,
                            resolution=0.01,
                            variable=current_min_brightness,
                            command=slider_min_changed,
                            length=300)
    
    label_1 = Tk.Label(root, text="Max Brightness")
    label_2 = Tk.Label(root, text="Min Brightness")

    label_1.pack(side=Tk.TOP)
    scale_max.pack(side=Tk.TOP)
    label_2.pack(side=Tk.TOP)
    scale_min.pack(side=Tk.TOP)

    projector.set_up_window()

    while True:
        projector.project_pattern(patterns[pattern_num][0])
        
        if cameras[0].type == 'web':
            _1 = cameras[0].get_image()
        if cameras[1].type == 'web':
            _2 = cameras[1].get_image()
        
        frame_1 = cameras[0].get_image()
        frame_2 = cameras[1].get_image()

        if cameras[0].type == 'web':
            frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
        if cameras[1].type == 'web':
            frame_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)
        
        a1 = f.add_subplot(221)
        a1.plot(frame_1[520, :])
        a1.set_ylim((0, 255))
        b1 = f.add_subplot(222)
        b1.imshow(frame_1)

        a2 = f.add_subplot(223)
        a2.plot(frame_2[520, :])
        a2.set_ylim((0, 255))
        b2 = f.add_subplot(224)
        b2.imshow(frame_2)
        root.update()

        if (end):
            # Save results of calibration
            with open('config.json') as f:
                data = json.load(f)

            data['projector']["min_brightness"] = projector.min_image_brightness
            data['projector']["max_brightness"] = projector.max_image_brightness

            with open('config.json', 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            projector.close_window()

            break
        canvas.draw()
        f.delaxes(a1)
        f.delaxes(b1)
        f.delaxes(a2)
        f.delaxes(b2)