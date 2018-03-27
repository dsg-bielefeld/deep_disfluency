from watson_streaming import transcribe
from incremental_asr import ASR
from copy import deepcopy


class IBMWatsonASR(ASR):

    def __init__(self, credentials_file=None, new_hypothesis_callback=None,
                 settings=None):
        ASR.__init__(self, credentials_file,  new_hypothesis_callback,
                     settings)
        # Provide a dictionary of Watson input and output features.
        # For example
        if not settings:
            self.settings = {
                'inactivity_timeout': -1,  # Don't kill me after 30 seconds
                'interim_results': True,
                'timestamps': True
            }
        else:
            self.settings = settings

    def clean_word(self, word):
        return 'uh-huh' if word in ['mmhm', 'aha', 'uhhuh'] else \
            'uh' if word in ["%HESITATION"] else word

    # Write whatever you want in your callback function (expecting a dict)
    def callback(self, data):
        if 'results' in data:
            transcript = data['results'][0]['alternatives'][0]['transcript']
            top_diff = data['results'][0]['alternatives'][0]['timestamps']
            word_diff, rollback = self.get_diff_and_update_word_graph(top_diff)
            word_diff = [(self.clean_word(h[0]),
                          h[1], h[2])
                         for h in word_diff]
            # print "ASR input", word_diff, rollback
            # for h in self.word_graph:
            #    print h
            if self.new_hypothesis_callback:
                self.new_hypothesis_callback(word_diff, rollback,
                                             self.word_graph)

    def listen(self):
        ASR.listen(self)
        # You can't ask for a simpler API than this!
        # nb will it continuously output the new diffs?
        transcribe(self.callback, self.settings, self.credentials_file)
