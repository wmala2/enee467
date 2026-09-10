import os
import socket

# Define the Network Information to Communicate with the rover - take a look at firmware UML diagram
# for more information. The address is whatever your firmware build was given, so it is settable
# rather than fixed: ROVER_IP in the environment, or set_rover_ip() from an entry point.
DEFAULT_UDP_IP = "192.168.50.223"
UDP_IP = os.environ.get("ROVER_IP", DEFAULT_UDP_IP)
UDP_PORT = 9000
UDP_REPLY_PORT = (
    9001  # firmware sends encoder/lidar replies here; owned by EncoderPoller when active
)

network_name = "BaleNet"
network_password = "F1ockOfTurtle$"

# Create the UDP Gateway
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def set_rover_ip(ip):
    """Point every later send at this rover. send_message reads the module global each call,
    so existing callers pick it up without being changed."""
    global UDP_IP
    if not ip:
        raise ValueError("rover ip must be a non-empty address")
    UDP_IP = ip


# Send our message over in JSON format
def send_message(msg_tuple):
    sock.sendto(msg_tuple, (UDP_IP, UDP_PORT))
