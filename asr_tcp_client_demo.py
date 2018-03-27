"""
A TCP client that listen to the computer sound card (e.g. microphone),
sends raw audio samples to a TCP server (implemented in
`asr_tcp_server_demo.py`) and receives disfluency tags.
"""
from __future__ import print_function

import socket
import threading

import sounddevice

HOST = '127.0.0.1'
PORT = 50007
AUDIO_OPTS = {
    'dtype': 'int16',
    'samplerate': 44100,
    'channels': 1,
}
BUFFER_SIZE = 2048


def tcp_receiver(sock):
    print('Receiving...')
    while True:
        print(sock.recv(1024), end='')


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))
print('Connected')

t = threading.Thread(target=tcp_receiver, args=(sock, ))
t.daemon = True
t.start()

with sounddevice.RawInputStream(**AUDIO_OPTS) as stream:
    while True:
        chunk, _ = stream.read(BUFFER_SIZE)
        sock.sendall(chunk[:])
s.close()
