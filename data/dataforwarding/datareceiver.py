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

This is the code on the Raspberry pi that allows it to receive information from a host computer

"""

import socket

# Define the IP address and port to listen on
raspberry_pi_ip = "0.0.0.0"  # Listen on all available interfaces
raspberry_pi_port = 5005

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind the socket to the IP and port
sock.bind((raspberry_pi_ip, raspberry_pi_port))

print(f"Listening on {raspberry_pi_ip}:{raspberry_pi_port}") # may need to be changed for python2

while True:
    # Receive data from the client
    data, addr = sock.recvfrom(1024)  # Buffer size is 1024 bytes
    received_data = data.decode('utf-8')
    
    print("Received: " + received_data)

# Note: This script will run indefinitely, receiving and printing data until manually stopped.
