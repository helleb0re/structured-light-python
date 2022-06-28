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

import numpy as np
import cv2


def calibration_patterns(patterns, cameras, projector):
    end = False

    def quit(root):
        nonlocal end
        end = True
        root.quit()     # Stops mainloop
        root.destroy()  # This is necessary on Windows to prevent
                        # Fatal Python Error: PyEval_RestoreThread: NULL tstate

    def on_key_event(event, canvas, toolbar):
        print('you pressed %s' % event.key)
        if (event.key == 'j'):
            projector.max_image_brightness += 1
        elif (event.key == 'k'):
            projector.min_image_brightness += 1
        elif (event.key == 'n'):
            projector.max_image_brightness -= 1
        elif (event.key == 'm'):
            projector.min_image_brightness -= 1
        key_press_handler(event, canvas, toolbar)

    root = Tk.Tk()
    root.wm_title("Embedding in TK")

    f = Figure(figsize=(5, 4), dpi=100)

    # A tk.DrawingArea
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

    canvas.mpl_connect('key_press_event', lambda event: on_key_event(event, canvas, toolbar))

    button = Tk.Button(master=root, text='Quit', command=lambda: quit(root))
    button.pack(side=Tk.BOTTOM)

    while True:
        response = next(projector.project_patterns(patterns))
        # for _ in projector.projection(new_pattern.astype(np.uint8)):
        if response:
            # cv2.imshow(window_name, new_pattern.astype(np.uint8))
            print(f'min: {projector.min_image_brightness}; max: {projector.max_image_brightness};')
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
            m, n, _ = frame_1.shape
            a1 = f.add_subplot(221)
            a1.plot(range(n), frame_1[520, :])
            b1 = f.add_subplot(222)
            b1.imshow(frame_1)

            a2 = f.add_subplot(223)
            a2.plot(range(n), frame_2[520, :])
            b2 = f.add_subplot(224)
            b2.imshow(frame_2)
            root.update()
            if (end):
                with open('config.json') as f:
                    data = json.load(f)
                data['projector']["min_brightness"] = projector.min_image_brightness
                data['projector']["max_brightness"] = projector.max_image_brightness
                with open('config.json', 'w') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                cv2.destroyAllWindows()
                break
            canvas.draw()
            f.delaxes(a1)
            f.delaxes(b1)
            f.delaxes(a2)
            f.delaxes(b2)   
