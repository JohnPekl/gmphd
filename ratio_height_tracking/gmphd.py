simplesum = sum  # we want to be able to use "pure" sum not numpy (shoulda namespaced)
from numpy import *
import numpy.linalg
from copy import deepcopy
from operator import attrgetter
import uuid
from scipy.optimize import linear_sum_assignment
from functools import partial

myfloat = float64


class GmphdComponent:
    """Represents a single Gaussian component,
    with a float weight, vector location, matrix covariance.
    Note that we don't require a GM to sum to 1, since not always about proby densities."""

    def __init__(self, weight, loc, cov, id=None):
        self.weight = myfloat(weight)
        self.loc = array(loc, dtype=myfloat, ndmin=2)
        self.cov = array(cov, dtype=myfloat, ndmin=2)
        self.loc = reshape(self.loc, (size(self.loc), 1))  # enforce column vec
        self.cov = reshape(self.cov, (size(self.loc), size(self.loc)))  # ensure shape matches loc shape
        self.invcov = numpy.linalg.inv(self.cov)
        if id is None:
            self.id = uuid.uuid4().int
        else:
            self.id = id


# We don't always have a GmphdComponent object so:
def dmvnorm(loc, cov, x):
    "Evaluate a multivariate normal, given a location (vector) and covariance (matrix) and a position x (vector) at which to evaluate"
    # The multivariate normal distribution
    # f(x1,x2,...,xk) = exp(-1/2 * (x-mu).T * cov-1 * (x-mu)) / sqrt((2*pi)^k * det(cov))
    loc = array(loc, dtype=myfloat)
    cov = array(cov, dtype=myfloat)
    x = array(x, dtype=myfloat)
    k = len(loc)
    part1 = (2.0 * pi) ** (-k * 0.5)
    part2 = power(numpy.linalg.det(cov), -0.5)
    dev = x - loc
    part3 = exp(-0.5 * dot(dot(dev.T, numpy.linalg.inv(cov)), dev))
    return part1 * part2 * part3


################################################################################
class Gmphd:
    """Represents a set of modelling parameters and the latest frame's
       GMM estimate, for a GM-PHD model without spawning.

       Typical usage would be, for each frame of input data, to run:
          g.update(obs)
          g.prune()
          estimate = g.extractstates()

      'gmm' is an array of GmphdComponent items which makes up
           the latest GMM, and updated by the update() call.
           It is initialised as empty."""

    def __init__(self, birthgmm, survival, detection, f, q, h, r, clutter):
        """
          'birthgmm' is an array of GmphdComponent items which makes up
               the GMM of birth probabilities.
          'survival' is survival probability.
          'detection' is detection probability.
          'f' is state transition matrix F.
          'q' is the process noise covariance Q.
          'h' is the observation matrix H.
          'r' is the observation noise covariance R.
          'clutter' is the clutter intensity.
          """
        self.gmm = []  # empty - things will need to be born before we observe them
        self.birthgmm = birthgmm
        self.survival = myfloat(survival)  # p_{s,k}(x) in paper
        self.detection = myfloat(detection)  # p_{d,k}(x) in paper
        self.f = array(f, dtype=myfloat)  # state transition matrix      (F_k-1 in paper)
        self.q = array(q, dtype=myfloat)  # process noise covariance     (Q_k-1 in paper)
        self.h = array(h, dtype=myfloat)  # observation matrix           (H_k in paper)
        self.r = array(r, dtype=myfloat)  # observation noise covariance (R_k in paper)
        self.clutter = myfloat(clutter)  # clutter intensity (KAU in paper)

        self.track_id = 0
        self.pre_state = []

    def update(self, obs):
        """Run a single GM-PHD step given a new frame of observations.
          'obs' is an array (a set) of this frame's observations.
          Based on Table 1 from Vo and Ma paper."""
        #######################################
        # Step 1 - prediction for birth targets
        born = [deepcopy(comp) for comp in self.birthgmm]
        # The original paper would do a spawning iteration as part of Step 1.
        spawned = []  # not implemented

        #######################################
        # Step 2 - prediction for existing targets
        updated = [GmphdComponent(self.survival * comp.weight, dot(self.f, comp.loc),
                                  self.q + dot(dot(self.f, comp.cov), self.f.T), comp.id)
                   for comp in self.gmm]

        predicted = born + spawned + updated

        #######################################
        # Step 3 - construction of PHD update components
        # These two are the mean and covariance of the expected observation
        nu = [dot(self.h, comp.loc) for comp in predicted]
        s = [self.r + dot(dot(self.h, comp.cov), self.h.T) for comp in predicted]
        # Not sure about any physical interpretation of these two...
        k = [dot(dot(comp.cov, self.h.T), linalg.inv(s[index]))
             for index, comp in enumerate(predicted)]
        pkk = [dot(eye(len(k[index])) - dot(k[index], self.h), comp.cov)
               for index, comp in enumerate(predicted)]

        #######################################
        # Step 4 - update using observations
        # The 'predicted' components are kept, with a decay
        newgmm = [GmphdComponent(comp.weight * (1.0 - self.detection), comp.loc, comp.cov, comp.id)
                  for comp in predicted]

        # then more components are added caused by each obsn's interaction with existing component
        for anobs in obs:
            anobs = array(anobs)
            newgmmpartial = []
            for j, comp in enumerate(predicted):
                newgmmpartial.append(GmphdComponent(
                    self.detection * comp.weight * dmvnorm(nu[j], s[j], anobs),
                    comp.loc + dot(k[j], anobs - nu[j]), pkk[j]))

            # The Kappa thing (clutter and reweight)
            weightsum = simplesum(newcomp.weight for newcomp in newgmmpartial)
            reweighter = 1.0 / (self.clutter + weightsum)
            for newcomp in newgmmpartial:
                newcomp.weight *= reweighter

            newgmm.extend(newgmmpartial)

        self.gmm = newgmm

    def prune(self, truncthresh=1e-6, mergethresh=0.01, maxcomponents=100):
        """Prune the GMM. Alters model state.
          Based on Table 2 from Vo and Ma paper."""
        # Truncation is easy
        weightsums = [simplesum(comp.weight for comp in self.gmm)]  # diagnostic
        sourcegmm = list(filter(lambda comp: comp.weight > truncthresh, self.gmm))
        weightsums.append(simplesum(comp.weight for comp in sourcegmm))
        origlen = len(self.gmm)
        trunclen = len(sourcegmm)
        # Iterate to build the new GMM
        newgmm = []
        while len(sourcegmm) > 0:
            # find weightiest old component and pull it out
            windex = argmax(comp.weight for comp in sourcegmm)
            weightiest = sourcegmm[windex]
            sourcegmm = sourcegmm[:windex] + sourcegmm[windex + 1:]
            # find all nearby ones and pull them out
            distances = [float(dot(dot((comp.loc - weightiest.loc).T, comp.invcov), comp.loc - weightiest.loc)) for comp
                         in sourcegmm]
            dosubsume = array([dist <= mergethresh for dist in distances])
            subsumed = [weightiest]
            if any(dosubsume):
                # print("Subsuming the following locations into weightest with loc %s and weight %g (cov %s):" \
                #	% (','.join([str(x) for x in weightiest.loc.flat]), weightiest.weight, ','.join([str(x) for x in weightiest.cov.flat]))
                # print(list([comp.loc[0][0] for comp in list(array(sourcegmm)[ dosubsume]) ])
                subsumed.extend(list(array(sourcegmm)[dosubsume]))
                sourcegmm = list(array(sourcegmm)[~dosubsume])
            # create unified new component from subsumed ones
            aggweight = simplesum(comp.weight for comp in subsumed)
            newcomp = GmphdComponent(aggweight,
                                     sum(array([comp.weight * comp.loc for comp in subsumed]), 0) / aggweight,
                                     sum(array([comp.weight * (
                                             comp.cov + (weightiest.loc - comp.loc) * (weightiest.loc - comp.loc).T)
                                                for comp in subsumed]), 0) / aggweight,
                                     weightiest.id)
            newgmm.append(newcomp)

        # Now ensure the number of components is within the limit, keeping the weightiest
        newgmm.sort(key=attrgetter('weight'))
        newgmm.reverse()
        self.gmm = newgmm[:maxcomponents]
        weightsums.append(simplesum(comp.weight for comp in newgmm))
        weightsums.append(simplesum(comp.weight for comp in self.gmm))
        print("prune(): %i -> %i -> %i -> %i" % (origlen, trunclen, len(newgmm), len(self.gmm)))
        print("prune(): weightsums %g -> %g -> %g -> %g" % (weightsums[0], weightsums[1], weightsums[2], weightsums[3]))
        # pruning should not alter the total weightsum (which relates to total num items) - so we renormalise
        weightnorm = weightsums[0] / weightsums[3]
        for comp in self.gmm:
            comp.weight *= weightnorm

    def extractstates(self, bias=1.0):
        """Extract the multiple-target states from the GMM.
          Returns a list of target states; doesn't alter model state.
          Based on Table 3 from Vo and Ma paper.
          I added the 'bias' factor, by analogy with the other method below."""
        items = []
        print("weights:")
        print([round(comp.weight, 7) for comp in self.gmm])
        for comp in self.gmm:
            val = comp.weight * float(bias)
            if val > 0.5:
                for _ in range(int(round(val))):
                    items.append(deepcopy(comp.loc))
        for x in items: print(x.T)
        return items

    def extractstatesusingintegral(self, bias=1.0):
        """Extract states based on the expected number of states from the integral of the intensity.
        This is NOT in the GMPHD paper; added by Dan.
        "bias" is a multiplier for the est number of items.
        """
        numtoadd = int(round(float(20000) * simplesum(comp.weight for comp in self.gmm)))
        if numtoadd > len(self.gmm):
            numtoadd = len(self.gmm)
        print("bias is %g, numtoadd is %i" % (bias, numtoadd))
        items = []
        # A temporary list of peaks which will gradually be decimated as we steal from its highest peaks
        peaks = [{'loc': comp.loc, 'weight': comp.weight, 'id': comp.id} for comp in self.gmm]
        while numtoadd > 0:
            windex = 0
            wsize = 0
            for which, peak in enumerate(peaks):
                if peak['weight'] > wsize:
                    windex = which
                    wsize = peak['weight']
            # add the winner
            items.append([deepcopy(peaks[windex]['loc']), 0, peaks[windex]['id']])
            #peaks[windex]['weight'] -= 100.0
            peaks.pop(windex)
            numtoadd -= 1

        lp, lc = len(self.pre_state), len(items)  # pre_state and items is current state
        cost = numpy.ones([lp, lc]) * 100000000
        for i in range(0, lp):
            for j in range(0, lc):
                if (self.pre_state[i][2] == items[j][2]):
                    xp, yp, rp, hp, _, _, _, _ = self.pre_state[i][0]
                    xc, yc, rc, hc, _, _, _, _ = items[j][0]
                    wp, wc = rp*hp, rc*hc
                    bboxA = [xp-wp/2, hp-hp/2, wp, hp]
                    bboxB = [xc - wc / 2, hc - hc / 2, wc, hc]
                    cost[i, j] = self.bb_intersection_over_union(bboxA, bboxB) #sqrt((xp - xc) ** 2 + (yp - yc) ** 2)
        row_ind, col_ind = linear_sum_assignment(cost, maximize=False)
        for i, idx in enumerate(col_ind):
            items[idx][1] = self.pre_state[row_ind[i]][1]
        for i in range(0, lc):
            if i not in col_ind:
                self.track_id += 1
                items[i][1] = self.track_id

        self.pre_state = deepcopy(items)

        return items

    ########################################################################################
    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    def update_obs_mp(self, anobs, predicted, nu, s, pkk, k):
        newgmmpartial = []
        for j, comp in enumerate(predicted):
            newgmmpartial.append(GmphdComponent(
                self.detection * comp.weight * dmvnorm(nu[j], s[j], anobs),
                comp.loc + dot(k[j], anobs - nu[j]), pkk[j]))

        # The Kappa thing (clutter and reweight)
        weightsum = simplesum(newcomp.weight for newcomp in newgmmpartial)
        reweighter = 1.0 / (self.clutter + weightsum)
        for newcomp in newgmmpartial:
            newcomp.weight *= reweighter
        return newgmmpartial

    def update_mp(self, obs, pool):
        """Run a single GM-PHD step given a new frame of observations.
          'obs' is an array (a set) of this frame's observations.
          Based on Table 1 from Vo and Ma paper."""
        #######################################
        # Step 1 - prediction for birth targets
        born = [deepcopy(comp) for comp in self.birthgmm]
        # The original paper would do a spawning iteration as part of Step 1.
        spawned = []  # not implemented

        #######################################
        # Step 2 - prediction for existing targets
        updated = [GmphdComponent(self.survival * comp.weight, dot(self.f, comp.loc),
                                  self.q + dot(dot(self.f, comp.cov), self.f.T), comp.id)
                   for comp in self.gmm]

        predicted = born + spawned + updated

        #######################################
        # Step 3 - construction of PHD update components
        # These two are the mean and covariance of the expected observation
        nu = [dot(self.h, comp.loc) for comp in predicted]
        s = [self.r + dot(dot(self.h, comp.cov), self.h.T) for comp in predicted]
        # Not sure about any physical interpretation of these two...
        k = [dot(dot(comp.cov, self.h.T), linalg.inv(s[index]))
             for index, comp in enumerate(predicted)]
        pkk = [dot(eye(len(k[index])) - dot(k[index], self.h), comp.cov)
               for index, comp in enumerate(predicted)]

        #######################################
        # Step 4 - update using observations
        # The 'predicted' components are kept, with a decay
        newgmm = [GmphdComponent(comp.weight * (1.0 - self.detection), comp.loc, comp.cov, comp.id)
                  for comp in predicted]

        # then more components are added caused by each obsn's interaction with existing component
        result = pool.map_async(partial(self.update_obs_mp, predicted=predicted, nu=nu, s=s, pkk=pkk, k=k), obs)
        result = result.get()
        for newgmmpartial in result:
            newgmm.extend(newgmmpartial)

        self.gmm = newgmm
