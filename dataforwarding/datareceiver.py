import socket
import RPi.GPIO as GPIO
import time

# Set up the GPIO pin numbering mode
GPIO.setmode(GPIO.BCM)

# Set the GPIO pin numbers for the PWMs
pwm_left = 12
pwm_right = 13

# Set the GPIO pin numbers for non-PWMs
gpio_pin1 = 23
gpio_pin2 = 24

# Set up the GPIO as an output
GPIO.setup(pwm_left, GPIO.OUT)
GPIO.setup(pwm_right, GPIO.OUT)
GPIO.setup(gpio_pin1, GPIO.OUT)
GPIO.setup(gpio_pin2, GPIO.OUT)

# Set up PWM on the GPIO pin with a frequency of 1000Hz
pwm1 = GPIO.PWM(pwm_left, 1000)
pwm2 = GPIO.PWM(pwm_right, 1000)
pwm1.start(0)
pwm2.start(0)

# Define the duty_cycles for the PWM output pins on the pi
duty_cycle1 = 0
duty_cycle2 = 0

Stream_on = True

# Define the IP address and port to listen on
raspberry_pi_ip = "0.0.0.0" # Listen on all availabel interfaces
raspberry_pi_port = 5005
MAX_SPEED = 60

# Crease a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

#Bind the socket to the IP and port
sock.bind((raspberry_pi_ip, raspberry_pi_port))

print("Listening on " + str(raspberry_pi_ip) + ":" + str(raspberry_pi_port))

GPIO.output(gpio_pin1, GPIO.LOW)
GPIO.output(gpio_pin2, GPIO.LOW)

try:
    while Stream_on:
        # Receive data from client
        data, addr = sock.recvfrom(1024) # Buffer size is 1024 bytes
        received_data = data.decode('utf-8')

        x_coor, y_coor = received_data.split(",")
        x_coor, y_coor = float(x_coor), float(y_coor)
        lws = y_coor + x_coor
        rws = -1 * (y_coor - x_coor)
        try:
            if (lws < 0):
                speed = max(min( (1 + lws) * MAX_SPEED , 100), 0)
                pwm1.ChangeDutyCycle( speed )
                GPIO.output( gpio_pin1, GPIO.HIGH )
            else:
                speed = max(min( ( lws ) * MAX_SPEED , 100), 0)
                pwm1.ChangeDutyCycle( speed )
                GPIO.output( gpio_pin1, GPIO.LOW )
            if (rws < 0):
                speed = max(min( (1 + rws) * MAX_SPEED , 100), 0)
                pwm2.ChangeDutyCycle( speed )
                GPIO.output( gpio_pin2, GPIO.HIGH )
            else: 
                speed = max(min( (1 + rws) * MAX_SPEED , 100), 0)
                pwm2.ChangeDutyCycle( rws * MAX_SPEED )
                GPIO.output( gpio_pin2, GPIO.LOW )  
        except ValueError as e:
            print(e)
        time.sleep(0.1)
finally:
    # stop PWM
    pwm1.stop()
    pwm2.stop()

    # Clean up hte GPIO settings
    GPIO.cleanup()
    sock.close()
