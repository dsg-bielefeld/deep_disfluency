"""
A TCP server that receives raw audio and reply with disfluency tags.
"""
from __future__ import print_function

import socket
import threading

from deep_disfluency.asr.ibm_watson import IBMWatsonASR
from deep_disfluency.tagger.deep_tagger import DeepDisfluencyTagger
from deep_disfluency.tagger.pool import Pool
import watson_streaming

HOST = ''  # Symbolic name meaning all available interfaces
PORT = 50007  # Arbitrary non-privileged port
TCP_INPUT_BUFFER_SIZE = 1024  # Number of bytes to read from TCP socket
POOL_SIZE = 4  # How many taggers to instantiate
TAGGER_SETTINGS = {
    'config_file': 'deep_disfluency/experiments/experiment_configs.csv',
    'config_number': 35,
    'saved_model_dir': 'deep_disfluency/experiments/035/epoch_6',
    'use_timing_data': True,
}
WATSON_SETTINGS = {
    'inactivity_timeout': -1,  # Don't kill me after 30 seconds
    'interim_results': True,
    'timestamps': True
}


class TranscribeAdapter(object):
    """
    An adapter between a TCP connection, IBM Watson, and a disfluency tagger.
    It exposes `audio_gen` and `callback` methods to use with
    `watson_streaming.transcribe`.
    """

    def __init__(self, conn, tagger, input_buffer_size):
        self.conn = conn
        self.disf = tagger
        self.input_buffer_size = input_buffer_size
        self.asr = IBMWatsonASR(new_hypothesis_callback=self._new_word_hypotheses_handler)

    def audio_gen(self):
        """
        Generates raw audio samples from data received from the TCP connection.
        """
        while True:
            yield self.conn.recv(self.input_buffer_size)

    def callback(self, data):
        """
        Sends disfluency analysis over TCP connection.
        """
        return self.asr.callback(data)

    def _new_word_hypotheses_handler(self, word_diff, rollback, word_graph):
        """
        Define what you want to happen when new word hypotheses come in.
        Based on `asr_demo.py`.
        """
        if word_diff == []:
            return
        self.disf.rollback(rollback)
        last_end_time = 0
        if len(word_graph) > 1:
            last_end_time = word_graph[(len(word_graph)-len(word_diff))-1][-1]
        # tag new words and work out where the new tuples start
        new_output_start = max([0, len(self.disf.get_output_tags())-1])
        for word, _, end_time in word_diff:
            timing = end_time - last_end_time
            new_tags = self.disf.tag_new_word(word, timing=timing)
            start_test = max([0, len(self.disf.get_output_tags())-len(new_tags)-1])
            if start_test < new_output_start:
                new_output_start = start_test
            last_end_time = end_time
        # Send out all the output tags from the new word diff onwards
        for w, h in zip(
                word_graph[new_output_start:],
                self.disf.get_output_tags(with_words=False)[new_output_start:]):
            self.conn.sendall("{},{},{},{}\n".format(w[0], w[1], w[2], h))


class TCPHandler(threading.Thread):
    def __init__(self, conn, pool):
        super(TCPHandler, self).__init__()
        self.conn = conn
        self.pool = pool
        self.daemon = True

    def run(self):
        tagger = self.pool.checkout()
        try:
            adapter = TranscribeAdapter(self.conn, tagger, TCP_INPUT_BUFFER_SIZE)
            watson_streaming.transcribe(
                adapter.callback,
                WATSON_SETTINGS,
                "credentials.json",
                adapter.audio_gen,
            )
        finally:
            self.pool.checkin(tagger)


def main():
    pool = Pool(pool_size=POOL_SIZE, tagger_class=DeepDisfluencyTagger,
                init_kwargs=TAGGER_SETTINGS)
    print('Taggers pool ready')

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST, PORT))
    sock.listen(1)
    print('Server listening')

    while True:
        conn, addr = sock.accept()
        print('Connected by', addr)
        handler = TCPHandler(conn, pool)
        handler.start()


if __name__ == '__main__':
    main()
