"""
Taggers initialisation is expensive. Having a pool of taggers instantiated in
parallel make checking out and in fast. It's up to the client to make sure not
to exhaust the pool.
"""
import threading


class Pool(object):
    def __init__(self, pool_size, tagger_class, init_args=(), init_kwargs=None):
        """Instantiate the pool and its taggers."""

        if init_kwargs is None:
            init_kwargs = {}

        self.pool_size = pool_size
        self.tagger_class = tagger_class
        self.init_args = init_args
        self.init_kwargs = init_kwargs

        self._pool = set()
        self._pool_lock = threading.Lock()

        # We can do it with threads because the taggers are already parallelized
        init_threads = []
        for _ in range(self.pool_size):
            t = threading.Thread(target=self._init_instance)
            t.start()
            init_threads.append(t)
        for t in init_threads:
            t.join()

    def _init_instance(self):
        instance = self.tagger_class(*self.init_args, **self.init_kwargs)
        with self._pool_lock:
            self._pool.add(instance)

    def checkout(self):
        """Get a tagger from the pool."""
        with self._pool_lock:
            return self._pool.pop()

    def checkin(self, tagger):
        """Return a tagger to the pool"""
        tagger.reset()
        with self._pool_lock:
            self._pool.add(tagger)
