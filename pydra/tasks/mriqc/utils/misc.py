from collections import OrderedDict
import logging


logger = logging.getLogger(__name__)


def _flatten_dict(indict):

    out_qc = {}
    for k, value in list(indict.items()):
        if not isinstance(value, dict):
            out_qc[k] = value
        else:
            for subk, subval in list(value.items()):
                if not isinstance(subval, dict):
                    out_qc["_".join([k, subk])] = subval
                else:
                    for ssubk, ssubval in list(subval.items()):
                        out_qc["_".join([k, subk, ssubk])] = ssubval
    return out_qc


BIDS_COMP = OrderedDict(
    [
        ("subject_id", "sub"),
        ("session_id", "ses"),
        ("task_id", "task"),
        ("acq_id", "acq"),
        ("rec_id", "rec"),
        ("run_id", "run"),
    ]
)
