import time
import unittest

from deep_disfluency.tagger.pool import Pool


class FakeSlowInitTagger(object):
    def __init__(self):
        time.sleep(0.1)


class FakeInitTagger(object):
    def __init__(self):
        self.reset_called = False

    def reset(self):
        self.reset_called = True


class TestPool(unittest.TestCase):

    def test_pool_parallelism(self):
        start = time.time()
        Pool(pool_size=10, tagger_class=FakeSlowInitTagger)
        end = time.time()
        # 10 parallel instantiates, each takes 0.1 sec, should take less than 0.2
        # secs.
        self.assertLess(end - start, 0.2)

    def test_checkout(self):
        p = Pool(pool_size=2, tagger_class=FakeInitTagger)
        tagger = p.checkout()
        self.assertIsInstance(tagger, FakeInitTagger)
        self.assertEqual(len(p._pool), 1)

    def test_checkin_returns_to_pool(self):
        p = Pool(pool_size=2, tagger_class=FakeInitTagger)
        tagger = p.checkout()
        self.assertEqual(len(p._pool), 1)
        p.checkin(tagger)
        self.assertEqual(len(p._pool), 2)

    def test_checkin_resets(self):
        p = Pool(pool_size=2, tagger_class=FakeInitTagger)
        tagger = p.checkout()
        self.assertFalse(tagger.reset_called)
        p.checkin(tagger)
        self.assertTrue(tagger.reset_called)


if __name__ == '__main__':
    unittest.main()
