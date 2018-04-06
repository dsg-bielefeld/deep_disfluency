import unittest

from deep_disfluency.tagger.deep_tagger import DeepDisfluencyTagger


class TestDeepDisfluencyTagger(unittest.TestCase):

    def test_tagging_a_sentence(self):

        disf = DeepDisfluencyTagger(
            config_file="deep_disfluency/experiments/experiment_configs.csv",
            config_number=35,
            saved_model_dir="deep_disfluency/experiments/035/epoch_6",
            use_timing_data=True
        )

        data = [
            # word     pos     timing  tag
            ("john",   "NNP",  0.33,   "<f/><tc/>"),
            ("likes",  "VBP",  0.33,   "<f/><cc/>"),
            ("uh",     "UH",   0.33,   "<e/><i/><cc/>"),
            ("loves",  "VBP",  0.33,   "<rps id=\"3\"/><cc/>"),
            ("mary",   "NNP",  0.33,   "<f/><ct/>"),
            ("yeah",   "UH",   2.00,   "<f/><tc/>"),
        ]

        for word, pos, timing, _ in data:
            disf.tag_new_word(word, pos=pos, timing=timing)

        for (_, _, _, tag), result in zip(data, disf.output_tags):
            self.assertEqual(result, tag)


if __name__ == '__main__':
    unittest.main()
