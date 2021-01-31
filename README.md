IMPLEMENTATION IS BASED ON THE FOLLOWING PAPER
========================
*Vo, B. N., & Ma, W. K. (2006). The Gaussian mixture probability hypothesis density filter. IEEE Transactions on signal processing, 54(11), 4091-4104*.

    TABLE I: PSEUDOCODE FOR THE GAUSSIAN MIXTURE PHD FILTER
    
    TABLE II: PRUNING FOR THE GAUSSIAN MIXTURE PHD FILTER
    + The Gaussian mixture PHD filter also suffers from computation problems associated with 
        the increasing number of Gaussian components as time progresses.
    + A simple pruning procedure can be used to reduce the number of Gaussian components propagated 
        to the next time step. 

Object identity (tracklet) was implemented by the following paper:

_Pham, N. T., Huang, W., & Ong, S. H. (2007, December). Maintaining track continuity in GMPHD filter. In 2007 6th International Conference on Information, Communications & Signal Processing (pp. 1-5). IEEE._

DIFFERENCES FROM VO & MA
========================

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
Install packages: `opencv-python`, `numpy`, `ffmpeg`.

Download MOT20 dataset from https://motchallenge.net/data/MOT20/.

Copy MOT20/test/MOT20-04 to the root folder of this source code.

Create `output` folder inside MOT20-04 to store output track images.

Run: `python demo_mot20.py`

Tracking Result:

![Alt Text](./MOT20-04/mot20.gif)

Video: https://www.youtube.com/watch?v=QB7tTMdKpGY&feature=youtu.be


LICENCE
=======

Refer to the original [gmphd](https://github.com/danstowell/gmphd) repository by [danstowell](https://github.com/danstowell)
