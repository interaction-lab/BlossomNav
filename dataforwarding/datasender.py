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

This is the code that sends information from local host to a raspberry pi

"""

import socket
import time
from dataforwarding.InvalidAddressError import *
import re

class datasender():
    def __init__(self, ip, port):
        InvalidAddressError.validate_ip(ip)
        InvalidAddressError.validate_port(port)

        # Define the IP address and port of the Raspberry Pi
        self._raspberry_pi_ip = ip  # Replace with the actual IP address of your Raspberry Pi
        self._raspberry_pi_port = port

        # Create a UDP socket
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def forward(self, information):
        # Convert the data to bytes
        data_to_send = f"{information}".encode('utf-8')

        # Send the data to the Raspberry Pi
        self._sock.sendto(data_to_send, (self._raspberry_pi_ip, self._raspberry_pi_port))

        print(f"Sent: {data_to_send}")

    def close(self):
        self._sock.close()
    
if __name__ == "__main__":
    temp_datasender = datasender("192.168.1.14", 5005)
    for i in range(100):
        temp_datasender.forward(str(i))
        time.sleep(1)
    temp_datasender.close()
