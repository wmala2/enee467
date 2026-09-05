import socket

# Define the Network Information to Communicate with the rover - take a look at firmware UML diagram for more information
UDP_IP = "192.168.50.223"
UDP_PORT = 9000
UDP_REPLY_PORT = (
    9001  # firmware sends encoder/lidar replies here; owned by EncoderPoller when active
)

network_name = "BaleNet"
network_password = "F1ockOfTurtle$"

# Create the UDP Gateway
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


# Send our message over in JSON format
def send_message(msg_tuple):
    sock.sendto(msg_tuple, (UDP_IP, UDP_PORT))
