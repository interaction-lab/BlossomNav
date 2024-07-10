import subprocess
import os
import threading
from getkey import getkey, keys

from datacollections.joystick import *

class StreamRecorder:
    def __init__(self, url, output_file="output.mp4", fps=30):
        self.url = url
        self.output_file = output_file
        self.fps = fps
        self.recording_process = None
        self.recording = False
        self.joystick = JoystickApp()
        self.joystick_process = None
        self.joystick_running = False

    def start_recording(self):
        path = "data/recordings"
        if not os.path.exists(path):
            os.makedirs(path)
        output_path = os.path.join(path, self.output_file)
        self.recording_process = subprocess.Popen([
            "ffmpeg",
            "-y",  # Overwrite output files without asking
            "-r", str(self.fps),  # Set the frame rate
            "-i", self.url,
            "-c:v", "libx264",  # Use the H.264 video codec
            "-preset", "veryfast",  # Use a faster encoding preset
            "-tune", "zerolatency",  # Tune for low latency
            "-bufsize", "8192k",  # Set buffer size for encoding
            "-f", "mp4",  # Format as MP4
            output_path
        ])
        print(f"Started recording from {self.url} to {output_path} at {self.fps} FPS")

    def stop_recording(self):
        if self.recording_process:
            self.recording_process.terminate()
            self.recording_process = None
            print("Stopped recording.")

    def start_joystick(self):
        if not self.joystick_running:
            self.joystick_running = True
            self.joystick.start()
            print("Joystick started.")

    def listen_for_keys(self):
        while True:
            key = getkey()
            if key == keys.CTRL_L:
                if not self.recording:
                    self.start_recording()
                    self.start_joystick()
                    self.recording = True

    def run(self):
        key_listener_thread = threading.Thread(target=self.listen_for_keys, daemon=True)
        key_listener_thread.start()
        key_listener_thread.join()

if __name__ == "__main__":
    # Replace this URL with the actual streaming URL you want to record from
    URL = "http://192.168.1.14:8081/"
    recorder = StreamRecorder(URL)
    recorder.run()
