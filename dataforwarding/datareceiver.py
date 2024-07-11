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
and register this information as actions on pi

"""

import socket
import RPi.GPIO as GPIO
import time
import math

# Set up the GPIO pin numbering mode
GPIO.setmode(GPIO.BCM)

# Set the GPIO pin number for the pi zero 2
led_pin1 = 12
led_pin2 = 13 

# Set up the PWM on the GPIO pin with a frequency of 1000Hz
GPIO.setup(led_pin1, GPIO.OUT)
GPIO.setup(led_pin2, GPIO.OUT)

# Set up the PWM on the GPIO pin with a frequency of 1000 Hz
pwm1 = GPIO.PWM(led_pin1, 1000)
pwm2 = GPIO.PWM(led_pin2, 1000)
pwm1.start(0)
pwm2.start(0)

# Variable for if stream is on or not
Stream_on = True

# Define IP address and port to listen on
raspberry_pi_ip = "0.0.0.0" # Listen to all availabel interfaces
raspberry_pi_port = 5005

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind the socket to the IP and port
sock.bind(raspberry_pi_ip, raspberry_pi_port)

print("Listening on " + str(raspberry_pi_ip) + ":" + str(raspberry_pi_port))

try:
    while Stream_on:
        # Recieve data from client
        data, addr = sock.recvfrom(1024) # Buffer size is 1024 bytes
        data = data.decode('utf-8')

        x_coor, y_coor = data.split(',')

        try:
            duty_cycle1 = 50 + int(x_coor) // 5
            duty_cycle2 = 50 + int(y_coor) // 5
            if 0 <= duty_cycle1 <= 100:
                pwm1.ChangeDutyCycle(duty_cycle1)
            else:
                continue

            if 0 <= duty_cycle2 <= 100:
                pwm2.ChangeDutyCycle(duty_cycle2)
            else:
                continue

            print(str(duty_cycle1) + "|" + str(duty_cycle2))

        except ValueError as ve:
            print(ve)
finally:
    #stop PWM
    pwm1.ChangeDutyCycle(0)
    pwm2.ChangeDutyCycle(0)

    # Clean the GPIO settings
    GPIO.cleanup()

    # close the socket
    sock.close()