# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)


class InlinePool:
    """
    Basic extension to a pool which supports no parallelism transparently.
    Takes ownership of the parameter pool.

    :param pool: a :class:`multiprocessing.Pool` or `None`
    """

    def __init__(self, pool):
        self._pool = pool

    def starmap(self, f, ls):
        if self._pool:
            return self._pool.starmap(f, ls)
        return [f(*x) for x in ls]

    def __del__(self):
        if self._pool:
            self._pool.close()
