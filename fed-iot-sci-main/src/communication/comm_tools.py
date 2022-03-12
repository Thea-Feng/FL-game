# Randy Xiao 2021-6-10

import socket


def get_ip():
    # get ip address of current device
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 8009))
        IP = s.getsockname()[0]
    finally:
        s.close()
    return IP

    