import zmq
import time
import numpy as np 

HEADER = 64
PORT = 2000
# SERVER = socket.gethostbyname(socket.gethostname())
# ADDR = (SERVER, PORT)
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"


def recieve_data():
    my_data = []
    context = zmq.Context()
    server = context.socket(zmq.SUB)
    server.connect(f"tcp://127.0.0.1:{PORT}")
    server.setsockopt_string(zmq.SUBSCRIBE, '')
    for i in range(0,100):
            my_data.append(server.recv_pyobj())
    with open(r'/Users/alireza/Desktop/quic/quic/examples/files/under3.txt', 'a') as fp:
            fp.write("%s\n" % my_data)      
    return(my_data)
    
recieve_data()

