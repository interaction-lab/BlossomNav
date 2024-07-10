from gui.record import *
from data.dataforwarding.datasender import datasender
from gui.record2 import *

val = 0

if val == 1:
    root = tk.Tk()
    app = VideoStreamApp(root, 'http://192.168.1.14:8081/')
    root.mainloop()
else: 
    x = StreamRecorder("http://192.168.1.14:8081/")
    x.run()
