import itertools
def overlap(read_a, read_b, min_length=1):
    start = 0
    while True:
        start = read_a.find(read_b[:min_length], start)
        if start == -1:
            return 0
        if read_b.startswith(read_a[start:]):
            return len(read_a)-start
        start += 1

def find_overlap_large(string_set):
    max_overlapping_length = 0
    read_a, read_b = None, None
    for a, b in itertools.permutations(string_set, 2):
        overlapping_length = overlap(a, b)
        if overlapping_length < max_overlapping_length:
            continue
        if  overlapping_length > max_overlapping_length:
            read_a, read_b = a, b
            max_overlapping_length = overlapping_length
    return read_a, read_b,max_overlapping_length

def shortestCommonSuperstring(string_set):
    read_a, read_b, olen = find_overlap_large(string_set)
    while olen > 0:
        index_a=string_set.index(read_a)
        del string_set[index_a]
        index_b=string_set.index(read_b)
        del string_set[index_b]
        string_set.append(read_a + read_b[olen:])
        read_a, read_b, olen = find_overlap_large(string_set)
    return ''.join(string_set)

