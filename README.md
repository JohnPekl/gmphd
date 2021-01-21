

DIFFERENCES FROM VO & MA
========================
Vo, B. N., & Ma, W. K. (2006). The Gaussian mixture probability hypothesis density filter. IEEE Transactions on signal processing, 54(11), 4091-4104.

There are some differences from the GM-PHD algorithm described in Vo & Ma's paper:

* I have not implemented "spawning" of new targets from old ones, since I don't 
  need it. It would be straightforward to add it - see the original paper.

* Weights are adjusted at the end of pruning, so that pruning doesn't affect
  the total weight allocation.

* I provide an alternative approach to state-extraction (an alternative to
  Table 3 in the original paper) which makes use of the integral to decide how
  many states to extract.


USAGE
=====

Download MOT20 dataset from https://motchallenge.net/data/MOT20/.

Copy MOT20/test/MOT20-04 to the root folder of this source code.

Run: `python demo_mot20.py`


LICENCE
=======

Refer to the original [gmphd](https://github.com/danstowell/gmphd) repositories by [danstowell](https://github.com/danstowell)
