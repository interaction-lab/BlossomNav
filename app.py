from datacollections.gui import *
from datacollections.streamer import *
from utils.utils import read_yaml

CONFIG_PATH = "config.yaml"
config = read_yaml(CONFIG_PATH)
streaming_url = config["streaming_url"]

gui = 0

if gui == 1:
    root = tk.Tk()
    app = VideoStreamApp(root, streaming_url)
    root.mainloop()
else: 
    x = StreamRecorder(streaming_url)
    x.run()
