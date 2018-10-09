import itertools

import fluteline
import time
import watson_streaming


class FakeIBMWatsonStreamer(fluteline.Producer):
    '''A fake streamer to simulate for testing, whilst offline
    '''

    def __init__(self, fake_stream):
        super(FakeIBMWatsonStreamer, self).__init__()
        self.fake_stream = fake_stream  # a list of updates to be fired

    def enter(self):
        print "starting fake streaming"

    def exit(self):
        print "finished fake streaming"

    def produce(self):
        if len(self.fake_stream) > 0:
            update = self.fake_stream.pop(0)
            self.output.put(update)


class Printer(fluteline.Consumer):
    def consume(self, msg):
        print(msg)


class IBMWatsonAdapter(fluteline.Consumer):
    '''
    A fluteline consumer-producer that receives transcription from
    :class:`watson_streaming.Transcriber` and prepare them to work
    with the deep_disfluency tagger.
    '''

    def enter(self):
        self.running_id = itertools.count()
        # Messages that went down the pipeline, indexed by start_time.
        self.memory = {}
        # Track when Watson commits changes to clear the memory.
        self.result_index = 0

    def consume(self, data):
        if 'results' in data:
            self.clear_memory_if_necessary(data)
            for t in data['results'][0]['alternatives'][0]['timestamps']:
                self.process_timestamp(t)

    def clear_memory_if_necessary(self, data):
        if data['result_index'] > self.result_index:
            self.memory = {}
            self.result_index = data['result_index']

    def process_timestamp(self, timestamp):
        word, start_time, end_time = timestamp
        word = self.clean_word(word)

        if self.is_new(start_time):
            id_ = next(self.running_id)
        else:
            id_ = self.get_id_if_update(start_time, end_time, word)

        if id_ is not None:
            msg = {
                'start_time': start_time,
                'end_time': end_time,
                'word': word,
                'id': id_
            }
            self.memory[start_time] = msg
            self.output.put(msg)

    def clean_word(self, word):
        if word in ['mmhm', 'aha', 'uhhuh']:
            return 'uh-huh'
        if word == '%HESITATION':
            return 'uh'
        return word

    def is_new(self, start_time):
        if len(self.memory.keys()) == 0:
            return True
        last_update = sorted(self.memory.keys(), reverse=True)[0]
        return start_time >= self.memory[last_update]['end_time']

    def get_id_if_update(self, start_time, end_time, word):
        """Returns the first id being updated.
        Removes/revokes the ids also implicitly being removed
        (i.e. the words chronologically after the update.
        If no update return None."""
        if self.memory.get(start_time):
            old_start_time = self.memory[start_time]['start_time']
            old_end_time = self.memory[start_time]['end_time']
            old_word = self.memory[start_time]['word']
            if (start_time, end_time, word) == (old_start_time,
                                                old_end_time, old_word):
                return None  # a repeated word
        update_id = None
        update_start_times_to_revoke = []
        for old_id in sorted(self.memory.keys(), reverse=True):
            if start_time >= self.memory[old_id]['end_time']:
                # we've found the update
                break
            update_start_times_to_revoke.append(old_id)
            update_id = self.memory[old_id]['id']
        for start_time in update_start_times_to_revoke:
            self.memory.pop(start_time, None)
        self.running_id = itertools.count(update_id + 1)  # set the counter
        return update_id


if __name__ == '__main__':
    fake_updates_raw_1 = [
        [('hello', 0, 1),
         ('my', 1, 2),
         ('name', 2, 3)
         ],

        [('hello', 0.5, 1),
         ('my', 1, 2),
         ('bame', 2, 3)
         ],

        [('once', 3.4, 4),
         ('upon', 4.2, 4.6),
         ('on', 4.3, 4.8)
         ]
    ]

    fake_updates_raw_2 = [
        # First new
        [
            ('hello', 0, 1),
        ],
        # Old and add new
        [
            ('hello', 0, 1),
            ('my', 1, 2),
        ],
        # Updating old timestamp and add new
        [
            ('hello', 0.5, 1),
            ('my', 1, 2),
            ('name', 3, 4),
        ],
        # Updating old word
        [
            ('hello', 0.5, 1),
            ('your', 1, 2),
            ('name', 3, 4),
        ],
        # Multiple old and new ones with timestamp overlap
        [
            ('once', 3.4, 4),
            ('upon', 4.2, 4.6),
            ('on', 4.3, 4.8),
        ]
    ]
    # create a fake list of incoming transcription result dicts from watson
    fake_updates_data = []
    result_index = 0
    for update in fake_updates_raw_2:
        data = {
            'result_index': result_index,
            'results': [{'alternatives': [{'timestamps': update}]}]
        }
        fake_updates_data.append(data)

    nodes = [
        FakeIBMWatsonStreamer(fake_updates_data),
        IBMWatsonAdapter(),
        Printer()
    ]

    tic = time.clock()

    fluteline.connect(nodes)
    fluteline.start(nodes)

    print time.clock() - tic, "seconds"

    time.sleep(1)
    fluteline.stop(nodes)
