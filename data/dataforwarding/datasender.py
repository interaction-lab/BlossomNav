import socket

# Define the IP address and port of the Raspberry Pi
raspberry_pi_ip = "192.168.1.100"  # Replace with the actual IP address of your Raspberry Pi
raspberry_pi_port = 5005

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Define the data to be sent (string and numerical data)
data_string = "Hello, Raspberry Pi!"
data_number = 42

# Convert the data to bytes
data_to_send = f"{data_string},{data_number}".encode('utf-8')

# Send the data to the Raspberry Pi
sock.sendto(data_to_send, (raspberry_pi_ip, raspberry_pi_port))

print(f"Sent: {data_to_send}")

# Close the socket
sock.close()
