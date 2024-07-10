"""
__________.__                                      _______               
\______   \  |   ____  ______ __________   _____   \      \ _____ ___  __
 |    |  _/  |  /  _ \/  ___//  ___/  _ \ /     \  /   |   \\__  \\  \/ /
 |    |   \  |_(  <_> )___ \ \___ (  <_> )  Y Y  \/    |    \/ __ \\   / 
 |______  /____/\____/____  >____  >____/|__|_|  /\____|__  (____  /\_/  
        \/                \/     \/            \/         \/     \/      

Copyright (c) 2024 Interactions Lab
License: MIT
Authors: Anthony Song and Nathan Dennler, Cornell University & University of Southern California
Project Page: https://github.com/interaction-lab/BlossomNav.git

This script is a joystick that will allow you to control the base for the Blossom

"""

import tkinter as tk
import math

from data.dataforwarding.datasender import datasender

class JoystickApp:
    def __init__(self):
        self.root = None
        self.canvas = None
        self.coord_label = None
        self.small_circle = None

    def start(self):
        if self.root is None:
            self.root = tk.Tk()
            self.root.title("2D Joystick")
            self.canvas = tk.Canvas(self.root, width=540, height=540, bg="black")
            self.canvas.pack()

            self.radius_big = 250
            self.radius_small = 40
            self.center_x = 270
            self.center_y = 270


            # Draw the big circle
            self.canvas.create_oval(self.center_x - self.radius_big, self.center_y - self.radius_big,
                                    self.center_x + self.radius_big, self.center_y + self.radius_big,
                                    outline="white", width=2)

            # Draw the small circle
            self.small_circle = self.canvas.create_oval(self.center_x - self.radius_small, self.center_y - self.radius_small,
                                                        self.center_x + self.radius_small, self.center_y + self.radius_small,
                                                        fill="white")

            # Add a label to display coordinates
            self.coord_label = tk.Label(self.root, text="Coordinates: (0, 0)", fg="white", bg="black", font=("Helvetica", 16))
            self.coord_label.pack()

            # Bind mouse events
            self.canvas.tag_bind(self.small_circle, "<Button-1>", self.on_click)
            self.canvas.tag_bind(self.small_circle, "<B1-Motion>", self.on_drag)
            self.canvas.tag_bind(self.small_circle, "<ButtonRelease-1>", self.on_release)

            self.drag_data = {"x": 0, "y": 0}
            self.is_dragging = False

            self.forwarder = datasender("192.168.1.14", 5005)

            # Start updating coordinates
            self.update_coordinates()

            # Run the Tkinter main loop
            self.root.mainloop()
    
    def stop(self):
        if self.root:
            self.forwarder.close()
            self.root.quit()  # Stop the Tkinter main loop
            self.root.destroy()  # Destroy the Tkinter window
            self.root = None

    def on_click(self, event):
        # Store the mouse drag data
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y
        self.is_dragging = True

    def on_drag(self, event):
        # Compute how much the mouse has moved
        dx = event.x - self.drag_data["x"]
        dy = event.y - self.drag_data["y"]

        # Update the drag data
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

        # Calculate new position
        new_x = self.canvas.coords(self.small_circle)[0] + dx
        new_y = self.canvas.coords(self.small_circle)[1] + dy
        new_x_center = new_x + self.radius_small
        new_y_center = new_y + self.radius_small

        # Calculate the distance from the center
        distance = math.sqrt((new_x_center - self.center_x) ** 2 + (new_y_center - self.center_y) ** 2)

        # Check if the new position is within the big circle
        if distance + self.radius_small <= self.radius_big:
            self.canvas.move(self.small_circle, dx, dy)
        else:
            # Calculate the angle and set the position at the edge of the big circle
            angle = math.atan2(new_y_center - self.center_y, new_x_center - self.center_x)
            edge_x = self.center_x + (self.radius_big - self.radius_small) * math.cos(angle)
            edge_y = self.center_y + (self.radius_big - self.radius_small) * math.sin(angle)
            self.canvas.coords(self.small_circle,
                               edge_x - self.radius_small, edge_y - self.radius_small,
                               edge_x + self.radius_small, edge_y + self.radius_small)

    def on_release(self, event):
        self.is_dragging = False
        self.snap_back_to_center()

    def snap_back_to_center(self):
        # Snap the small circle back to the center
        self.canvas.coords(self.small_circle,
                           self.center_x - self.radius_small, self.center_y - self.radius_small,
                           self.center_x + self.radius_small, self.center_y + self.radius_small)

    def update_coordinates(self):
        if self.root:
            # Get the current coordinates of the small circle
            coords = self.canvas.coords(self.small_circle)
            x_center = (coords[0] + coords[2]) / 2
            y_center = (coords[1] + coords[3]) / 2

            # Calculate the position relative to the origin
            x_relative = round(x_center - self.center_x)
            y_relative = round(self.center_y - y_center)

            # Update the label
            self.coord_label.config(text=f"Coordinates: ({x_relative}, {y_relative})")

            self.forwarder.forward(f"{x_relative},{y_relative}")

            # Schedule the next update
            self.root.after(100, self.update_coordinates)

if __name__ == "__main__":
    app = JoystickApp()
    app.start()
    app.stop()
