from __future__ import division
from deep_tagger import DeepDisfluencyTagger
import fluteline


class DeepTaggerModule(fluteline.Consumer):
    """A fluteline incremental concurrent Consumer module which
    consumes update increments as word dictionaries, e.g.:

            {'id': 1,
            'start_time': 0.44,
            'end_time': 0.77,
            'word': 'hello'}

    And produces the same update increments but with the POS tags
    and disfluency tags:

            {'id': 1,
            'start_time': 0.44,
            'end_time': 0.77,
            'word': 'john',
            'pos_tag' : 'NN'
            'disf_tag' : '<f/>'}

    These will be updated as diffs.
    """

    def __init__(self,
                 config_file="experiments/experiment_configs.csv",
                 config_number=35,
                 saved_model_dir="experiments/035/epoch_6",
                 use_timing_data=True):
        super(DeepTaggerModule, self).__init__()
        self.disf_tagger = DeepDisfluencyTagger(
            config_file=config_file,
            config_number=config_number,
            saved_model_dir=saved_model_dir,
            use_timing_data=use_timing_data
        )
        print "Deep Tagger Module ready"
        self.latest_word_ID = -1
        self.word_graph = []

    # def enter(self):

    def consume(self, word_update):
        """ Will get an update like:

            {'id': 1,
            'start_time': 0.44,
            'end_time': 0.77,
            'word': 'hello'}

        Add it to the tagger's word graph either at the end, or
        rolling back first and then add it.
        """
        try:
            # print "RECEIVING", word_update
            if word_update['id'] <= self.latest_word_ID:
                # rollback needed
                # TODO should be consistent
                backwards = (self.latest_word_ID - word_update['id']) + 1
                self.disf_tagger.rollback(backwards)
                self.word_graph = self.word_graph[:
                                                  len(self.word_graph) -
                                                  backwards]
            self.latest_word_ID = word_update['id']
            self.word_graph.append(word_update)
            timing = word_update['end_time'] - word_update['start_time']
            word = word_update['word']
            new_tags = self.disf_tagger.tag_new_word(word, timing=timing)
            start_id = self.latest_word_ID - (len(new_tags) - 1)
            word_update_indices = range(start_id, self.latest_word_ID + 1)
            # print "\nnew tags:"
            for idx, new_tag in zip(word_update_indices, new_tags):
                # update the disf tag and pos tag for new tag updates
                self.word_graph[idx]['disf_tag'] = new_tag
                pos_idx = idx + (self.disf_tagger.window_size - 1)
                self.word_graph[idx]['pos_tag'] = \
                    self.disf_tagger.word_graph[pos_idx][1]
                # print self.word_graph[idx]
                # output the new tags for the updated word
                self.output.put(self.word_graph[idx])
        except BaseException:
            print "Disfluency tagger failed to update with new word"
