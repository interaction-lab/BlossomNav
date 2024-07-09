import socket

# Define the IP address and port to listen on
raspberry_pi_ip = "0.0.0.0"  # Listen on all available interfaces
raspberry_pi_port = 5005

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind the socket to the IP and port
sock.bind((raspberry_pi_ip, raspberry_pi_port))

print(f"Listening on {raspberry_pi_ip}:{raspberry_pi_port}")

while True:
    # Receive data from the client
    data, addr = sock.recvfrom(1024)  # Buffer size is 1024 bytes
    received_data = data.decode('utf-8')
    
    # Split the received data into string and numerical parts
    data_string, data_number = received_data.split(',')
    data_number = int(data_number)
    
    print(f"Received from {addr}:")
    print(f"String: {data_string}")
    print(f"Number: {data_number}")

# Note: This script will run indefinitely, receiving and printing data until manually stopped.
