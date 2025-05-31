#!/usr/bin/env python3

import os
import numpy as np

# dataset: http://corpus-texmex.irisa.fr/

# taking from https://github.com/erikbern/ann-benchmarks/blob/master/ann_benchmarks/datasets.py#L164
def sift():
    import os
    import struct
    import tarfile
    from urllib.request import urlretrieve
    import numpy as np
    def _load_fn(t, fn):
        m = t.getmember(fn)
        f = t.extractfile(m)
        k, = struct.unpack("i", f.read(4))
        n = m.size // (4 + 4 * k)
        f.seek(0)
        return n, k, f
    def _load_fvecs(t, fn):
        n, k, f = _load_fn(t, fn)
        v = np.zeros((n, k), dtype=np.float32)
        for i in range(n):
            f.read(4)  # ignore vec length
            v[i] = struct.unpack("f" * k, f.read(k * 4))
        return v
    def _load_ivecs(t, fn):
        n, k, f = _load_fn(t, fn)
        v = np.zeros((n, k), dtype=np.int32)
        for i in range(n):
            f.read(4)  # ignore vec length
            v[i] = struct.unpack("i" * k, f.read(k * 4))
        return v
    # download dataset if we have not downloaded it yes
    url = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
    fn = os.path.join("./test_data", "sift.tar.gz")
    if not os.path.exists(fn):
        print(f"downloading {url} -> {fn}...")
        urlretrieve(url, fn)
    # load dataset
    with tarfile.open(fn, "r:gz") as t:
        train = _load_fvecs(t, "sift/sift_base.fvecs")
        test = _load_fvecs(t, "sift/sift_query.fvecs")
        neighbors = _load_ivecs(t, "sift/sift_groundtruth.ivecs") + 1
    print(f"train.shape: {train.shape}")
    print(f"test.shape: {test.shape}")
    print(f"neighbors.shape: {neighbors.shape}")
    return train, test, neighbors


'''
(base) root@di-20241115115906-kfh5w:~/code/sift1M # ll
total 563300
-rw-r--r--  1 51993 50100 516000000 Dec 15  2009 sift_base.fvecs
-rw-r--r--  1 51993 50100   4040000 Dec 15  2009 sift_groundtruth.ivecs
-rw-r--r--  1 51993 50100  51600000 Dec 15  2009 sift_learn.fvecs
-rw-r--r--  1 51993 50100   5160000 Dec 15  2009 sift_query.fvecs

file format:

Each comprises 3 subsets of vectors:
  • base vectors: the vectors in which the search is performed
  • query vectors
  • learning vectors: to find the parameters involved in a particular method

In addition, we provide the groundtruth for each set, in the form of the 
pre-computed k nearest neighbors and their square Euclidean distance. 

We use three different file formats: 
  • The vector files are stored in .bvecs or .fvecs format, 
  • The groundtruth file in is .ivecs format. 

.bvecs, .fvecs and .ivecs vector file formats:

The vectors are stored in raw little endian. 
Each vector takes 4+d*4 bytes for .fvecs and .ivecs formats, and 4+d bytes for .bvecs formats, 
where d is the dimensionality of the vector, as shown below. 

field 	 field type 	 description 
d	     int	         the vector dimension
components	(unsigned char|float | int)*d	the vector components

The only difference between .bvecs, .fvecs and .ivecs files is the base type for the vector components, which is unsigned char, float or int, respectively. 
'''

# return training vector, base vector, query vector, ground truth vector
def load_sift_data(data_path):
    if not os.path.exists(data_path):
        return None, None, None, None

    def ivecs_read(fname):
        a = np.fromfile(fname, dtype='int32')
        d = a[0]
        arr = a.reshape(-1, d + 1)[:, 1:].copy()
        print(f"{fname} data shape: {arr.shape}")
        return arr

    def fvecs_read(fname):
        return ivecs_read(fname).view('float32')

    xb = fvecs_read(f"{data_path}/sift_base.fvecs")
    xt = fvecs_read(f"{data_path}/sift_learn.fvecs")
    xq = fvecs_read(f"{data_path}/sift_query.fvecs")
    gt = ivecs_read(f"{data_path}/sift_groundtruth.ivecs")
    return xt, xb, xq, gt


'''
# python3 faiss/demos/demo_ondisk_ivf.py
sift1M/sift_base.fvecs data shape: (1000000, 128) # number of candidates: 1000000
sift1M/sift_learn.fvecs data shape: (100000, 128) # vector dimension: 128
sift1M/sift_query.fvecs data shape: (10000, 128)
sift1M/sift_groundtruth.ivecs data shape: (10000, 100) # number of knn neighbors: 100

# ls -lh /tmp/ | grep index
-rw-r--r-- 1 root root 127M May 27 14:39 block_0.index
-rw-r--r-- 1 root root 127M May 27 14:39 block_1.index
-rw-r--r-- 1 root root 127M May 27 14:39 block_2.index
-rw-r--r-- 1 root root 127M May 27 14:39 block_3.index
-rw-r--r-- 1 root root 496M May 27 14:40 merged_index.ivfdata
-rw-r--r-- 1 root root 2.1M May 27 14:40 populated.index
-rw-r--r-- 1 root root 2.1M May 27 14:36 trained.index
'''