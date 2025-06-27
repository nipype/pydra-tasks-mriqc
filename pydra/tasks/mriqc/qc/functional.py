import logging
import numpy as np
import os.path as op


logger = logging.getLogger(__name__)


def gsr(epi_data, mask, direction="y", ref_file=None, out_file=None):
    """
    Compute the :abbr:`GSR (ghost to signal ratio)` [Giannelli2010]_.

    The procedure is as follows:







    .. warning ::

      This should be used with EPI images for which the phase
      encoding direction is known.

    :param str epi_file: path to epi file
    :param str mask_file: path to brain mask
    :param str direction: the direction of phase encoding (x, y, all)
    :return: the computed gsr

    """
    direction = direction.lower()
    if direction[-1] not in ("x", "y", "all"):
        raise Exception(
            f"Unknown direction {direction}, should be one of x, -x, y, -y, all"
        )
    if direction == "all":
        result = []
        for newdir in ("x", "y"):
            ofile = None
            if out_file is not None:
                fname, ext = op.splitext(ofile)
                if ext == ".gz":
                    fname, ext2 = op.splitext(fname)
                    ext = ext2 + ext
                ofile = f"{fname}_{newdir}{ext}"
            result += [gsr(epi_data, mask, newdir, ref_file=ref_file, out_file=ofile)]
        return result
    # Roll data of mask through the appropriate axis
    axis = RAS_AXIS_ORDER[direction]
    n2_mask = np.roll(mask, mask.shape[axis] // 2, axis=axis)
    # Step 3: remove from n2_mask pixels inside the brain
    n2_mask = n2_mask * (1 - mask)
    # Step 4: non-ghost background region is labeled as 2
    n2_mask = n2_mask + 2 * (1 - n2_mask - mask)
    # Step 5: signal is the entire foreground image
    ghost = np.mean(epi_data[n2_mask == 1]) - np.mean(epi_data[n2_mask == 2])
    signal = np.median(epi_data[n2_mask == 0])
    return float(ghost / signal)


RAS_AXIS_ORDER = {"x": 0, "y": 1, "z": 2}
