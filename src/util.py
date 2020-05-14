import bisect

def sorted_insert(x, l: list, key=lambda x: x, lo=0, hi=None):
    """
    Inserts x into the supplied list l. The list l needs to be sorted from least to greatest.
    The key is a lambda that is called on the xs of the list to give them values by which
    they will be sorted. So, if we're sorting a list of genomes the key will probably be something
    like `lambda genome: genome.mse` which will use mean squared error to sort the genomes.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(l)
    while lo < hi:
        mid = (lo+hi)//2
        if key(l[mid]) < key(x): lo = mid+1
        else: hi = mid
    l.insert(lo, x)

    return lo
