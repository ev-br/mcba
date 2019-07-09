from __future__ import division, print_function, absolute_import

import itertools
import numpy as np
from numpy.testing import assert_allclose


def grouper(iterable, chunksize, f=tuple):
    """grouper(chunksize, iterable, f=tuple)
    Iterate over an iterable in chunks of chunksize.
    (c) Sven Marnach, http://stackoverflow.com/questions/8991506
    >>> for g in grouper([1, 2, 3, 4, 5], 2): print(g, '', end='')
    (1, 2) (3, 4) (5,) 
    >>> for g in grouper([1, 2, 3, 4, 5], 2, f = lambda x: sum(x)): 
    ...         print(g, '', end='')
    3 7 5 
    """
    it = iter(iterable)
    while True:
        chunk = f(itertools.islice(it, chunksize))
        if not chunk:
            return
        yield chunk


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ... (itertools recipes)"
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C  (itertools recipes)"
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = itertools.cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = itertools.cycle(itertools.islice(nexts, pending))



def copydict(basedict, dict2):
    """Useful for the walker_factory: 
    a list of dicts with slightly different parameters.
    """
    d = {}
    d.update(basedict, **dict2)
    return d


'''
##DEPRECATED in favor of the build-in fsum [2.6+]
################### priority queue based summations (an overkill?) ###########
import heapq

def heapsum(heap):
    """Priority queue based summation."""
    if len(heap) == 0:
        return 0

    while len(heap)>1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(heap, a+b)
    return heap[0]


def heapsum_chunked(it, chunksize=8196):
    """Given an iterable `it`, sum it up in a two-stage process:
    consume it in chunks, heapsum each chunk, then heapsum the chunked results.
    """
    heap_of_chunks = []
    for chunk in grouper(it, chunksize, list):
        heapq.heapify(chunk)
        s = heapsum(chunk)
        heapq.heappush(heap_of_chunks, s)
    return heapsum(heap_of_chunks)    
'''



#FIXME: assert -> return T/F
def compare_cnfs(cnf, cnf_1, check_roots=True, atol=1e-10, rtol=1e-10):
    """Given two cnfs, assert they coincide."""
    assert_allclose(cnf["FSfq"], cnf_1["FSfq"], atol=atol, rtol=rtol)                    
    assert_allclose(cnf["P"], cnf_1["P"], atol=atol, rtol=rtol) 
    if check_roots:   
        assert_allclose(cnf["roots"], cnf_1["roots"], atol=atol, rtol=rtol)
 



############### numpy structured array from a dict-like mapping ##########

def arr_from_dict(dct, datatype):
    """Construct a structured np.array from a dict-like mapping object, 
    using datatype as an np.dtype of the output array. 

    For example: 
    >>> dct = {"a": 1, "b": 2, "c": 3}
    >>> dt = np.dtype( [("a", "float64"), ("b", "float64")] )
    >>> arr_from_dict(dct, dt)
    array([( 1.,  2.)],
          dtype=[('a', '<f8'), ('b', '<f8')])

    Notice that dct["c"] is left out since there's no such key in the datatype.
    """
    tpl = tuple(dct[name] for name in datatype.names)
    return np.array([tpl], dtype=datatype) 
    


def arr_from_dictiter(it, datatype):
    """Construct a structured array from an iterable, using datatype as
    a dtype of the output array. 
    
    For example:
    >>> lst = [{"a": 1, "b": 2, "c": 3}, {"a": 4, "b": 5}]
    >>> dt = np.dtype( [("a", "float64"), ("b", "float64")] )
    >>> xx = arr_from_dictiter(lst, dt)
    >>> xx["a"], xx["b"]
    (array([ 1.,  4.]), array([ 2.,  5.]))
    """
    data = np.empty(1, dtype=datatype)
    for item in it: 
        data = np.r_[data, arr_from_dict(item, datatype)]
    data = np.delete(data, 0)
    return data

##############################
if __name__ == "__main__":
    import doctest
    doctest.testmod()
