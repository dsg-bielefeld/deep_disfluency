'''
Speech to text transcription, from your mic, in real-time,
using IBM Watson, then using the IBM Watson adapter and
Deep Disfluency module, detect disfluencies in real time.
'''

try:
    import deep_disfluency
except ImportError:
    print("no installed deep_disfluency package, pathing to source")
    import sys
    sys.path.append("../")

import argparse
import time

import fluteline

import watson_streaming
import watson_streaming.utilities
from deep_disfluency.asr.ibm_watson import IBMWatsonAdapter
from deep_disfluency.tagger.deep_tagger_module import DeepTaggerModule


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('credentials', help='path to credentials.json')
    return parser.parse_args()


class Printer(fluteline.Consumer):
    def consume(self, msg):
        print(msg)


def main():
    args = parse_arguments()
    settings = {
        'inactivity_timeout': -1,  # Don't kill me after 30 seconds
        'interim_results': True,
        'timestamps': True
    }

    nodes = [
        watson_streaming.utilities.MicAudioGen(),
        watson_streaming.Transcriber(settings, args.credentials),
        # watson_streaming.utilities.Printer(),
        IBMWatsonAdapter(),
        DeepTaggerModule(),
        Printer()

    ]

    fluteline.connect(nodes)
    fluteline.start(nodes)

    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        fluteline.stop(nodes)


if __name__ == '__main__':
    main()
