import itertools

import fluteline
import time
import watson_streaming


class FakeIBMWatsonStreamer(fluteline.Producer):
    '''A fake streamer to simulate for testing, whilst offline
    '''

    def __init__(self, fake_stream):
        super(fluteline.Producer, self).__init__()
        self.fake_stream = fake_stream  # a list of updates to be fired

    def enter(self):
        print "starting fake streaming"

    def exit(self):
        print "finished fake streaming"

    def produce(self):
        if len(self.fake_stream) > 0:
            update = self.fake_stream.pop(0)
            self.put(update)


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
        # Messages that went down the pipeline.
        self.memory = []
        # Track when Watson commits changes to clear the memory.
        self.result_index = 0

    def consume(self, data):
        if 'results' in data:
            self.clear_memory_if_necessary(data)
            for t in data['results'][0]['alternatives'][0]['timestamps']:
                self.process_timestamp(t)

    def clear_memory_if_necessary(self, data):
        if data['result_index'] > self.result_index:
            self.memory = []
            self.result_index = data['result_index']

    def process_timestamp(self, timestamp):
        word, start_time, end_time = timestamp
        word = self.clean_word(word)

        if self.is_new(start_time):
            id_ = next(self.running_id)
        else:
            old = self.get_message_to_update(word, start_time, end_time)
            if old:
                id_ = old['id']
                self.memory = [m for m in self.memory if m['id'] < id_]
                self.running_id = itertools.count(id_ + 1)
            else:
                id_ = None

        if id_ is not None:
            msg = {
                'start_time': start_time,
                'end_time': end_time,
                'word': word,
                'id': id_
            }
            self.memory.append(msg)
            self.put(msg)


    def clean_word(self, word):
        if word in ['mmhm', 'aha', 'uhhuh']:
            return 'uh-huh'
        if word == '%HESITATION':
            return 'uh'
        return word

    def is_new(self, start_time):
        if not self.memory:
            return True
        return start_time >= self.memory[-1]['end_time']

    def get_message_to_update(self, word, start_time, end_time):
        '''
        Returns the message from memory that the current one updates.
        If no update return None.
        '''
        for old in self.memory:
            # Search for overlap
            overlap_conditions = [
                old['start_time'] < start_time < old['end_time'],
                old['end_time'] < end_time < old['end_time'],
            ]
            if any(overlap_conditions):
                return old
            # Search for word change
            word_change_conditions = [
                old['start_time'] == start_time,
                old['end_time'] == end_time,
                old['word'] != word,
            ]
            if all(word_change_conditions):
                return old


if __name__ == '__main__':
    fake_updates_raw = [
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
    # create a fake list of incoming transcription result dicts from watson
    fake_updates_data = []
    result_index = 0
    for update in fake_updates_raw:
        data = {
            'result_index': result_index,
            'results': [{'alternatives': [{'timestamps': update}]}]
        }
        fake_updates_data.append(data)

    nodes = [
       FakeIBMWatsonStreamer(fake_updates_data),
       IBMWatsonAdapter(),
       Printer(),
    ]

    tic = time.clock()

    fluteline.connect(nodes)
    fluteline.start(nodes)

    print time.clock() - tic, "seconds"

    time.sleep(1)
    fluteline.stop(nodes)
