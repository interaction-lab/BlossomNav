from gui.gui import *

root = tk.Tk()
app = VideoStreamApp(root, 'http://192.168.1.14:8081/')
root.mainloop()