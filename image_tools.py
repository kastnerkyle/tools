# Author: Kyle Kastner
# License: BSD 3-Clause

import numpy as np
from scipy.linalg import eigh
from scipy.misc import imresize


def ind2sub(array_shape, ind):
    # Gives repeated indices, replicates matlabs ind2sub
    rows = (ind.astype("int32") // array_shape[1])
    cols = (ind.astype("int32") % array_shape[1])
    return (rows, cols)


def graphcut(im, n_splits=2, split_type="mean", rad=5, sigma_x=.3,
             sigma_p=.1, scaling=255.):
    # im: grayscale image
    sz = im.shape[0] * im.shape[1]
    ind = np.arange(sz)

    I, J = ind2sub(im.shape, ind)
    I = I + 1
    J = J + 1

    scaled_im = im.ravel() / float(scaling)

    # float32 gives the wrong answer...
    scaled_im = scaled_im.astype("float64")
    sim = np.zeros((sz, sz)).astype("float64")

    # Faster with broadcast tricks
    # Still wasting computation - einsum might be fastest
    x1 = I[None]
    x2 = I[:, None]
    y1 = J[None]
    y2 = J[:, None]
    dist = (x1 - x2) ** 2 + (y1 - y2) ** 2
    scale = np.exp(-(dist / (sigma_x ** 2)))
    sim = scale
    sim[np.sqrt(dist) >= rad] = 0.
    del x1
    del x2
    del y1
    del y2
    del dist

    p1 = scaled_im[None]
    p2 = scaled_im[:, None]
    pdist = (p1 - p2) ** 2
    pscale = np.exp(-(pdist / (sigma_p ** 2)))

    sim *= pscale

    dind = np.diag_indices_from(sim)
    sim[dind] = 1.

    d = np.sum(sim, axis=1)
    D = np.diag(d)
    A = (D - sim)

    # Want second smallest eigenvector onward
    S, V = eigh(A, D, eigvals=(1, n_splits + 1),
                overwrite_a=True, overwrite_b=True)
    sort_ind = np.argsort(S)
    S = S[sort_ind]
    V = V[:, sort_ind]
    segs = V
    segs[:, -1] = ind

    def cut(im, matches, ix, split_type="mean"):
        # Can choose how to split
        if split_type == "mean":
            split = np.mean(segs[:, ix])
        elif split_type == "median":
            split = np.median(segs[:, ix])
        elif split_type == "zero":
            split = 0.
        else:
            raise ValueError("Unknown split type %s" % split_type)

        meets = np.where(matches[:, ix] >= split)[0]
        match1 = matches[meets, :]
        res1 = np.zeros_like(im)
        match_inds = match1[:, -1].astype("int32")
        res1.ravel()[match_inds] = im.ravel()[match_inds]

        meets = np.where(matches[:, ix] < split)[0]
        match2 = matches[meets, :]
        res2 = np.zeros_like(im)
        match_inds = match2[:, -1].astype("int32")
        res2.ravel()[match_inds] = im.ravel()[match_inds]
        return (match1, match2), (res1, res2)

    # Recursively split partitions
    # Currently also stores intermediates
    all_splits = []
    all_matches = [[segs]]
    for i in range(n_splits):
        matched = all_matches[-1]
        current_splits = []
        current_matches = []
        for s in matched:
            matches, splits = cut(im, s, i, split_type=split_type)
            current_splits.extend(splits)
            current_matches.extend(matches)
        all_splits.append(current_splits)
        all_matches.append(current_matches)
    return all_matches, all_splits


def test_graphcut():
    import matplotlib.pyplot as plt
    from scipy.misc import lena
    im = lena()
    # Any bigger and my weak laptop gets memory errors
    bounds = (50, 50)
    im = imresize(im, bounds, interp="bicubic")
    all_matches, all_splits = graphcut(im, split_type="mean")

    to_plot = all_splits[-1]
    f, axarr = plt.subplots(2, len(to_plot) // 2)
    for n in range(len(to_plot)):
        axarr.ravel()[n].imshow(to_plot[n], cmap="gray")
        axarr.ravel()[n].set_xticks([])
        axarr.ravel()[n].set_yticks([])
    plt.show()

if __name__ == "__main__":
    test_graphcut()
