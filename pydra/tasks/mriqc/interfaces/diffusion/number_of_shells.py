import attrs
from fileformats.generic import File
import logging
import numpy as np
from pydra.compose import python
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV


logger = logging.getLogger(__name__)


@python.define
class NumberOfShells(python.Task["NumberOfShells.Outputs"]):
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.diffusion.number_of_shells import NumberOfShells

    """

    in_bvals: File
    b0_threshold: float = 50
    dsi_threshold: int = 11

    class Outputs(python.Outputs):
        models: list
        n_shells: int
        out_data: list
        b_values: list
        b_masks: list
        b_indices: list
        b_dict: dict

    @staticmethod
    def function(
        in_bvals: File, b0_threshold: float, dsi_threshold: int
    ) -> tuples[list, int, list, list, list, list, dict]:
        models = attrs.NOTHING
        n_shells = attrs.NOTHING
        out_data = attrs.NOTHING
        b_values = attrs.NOTHING
        b_masks = attrs.NOTHING
        b_indices = attrs.NOTHING
        b_dict = attrs.NOTHING
        in_data = np.squeeze(np.loadtxt(in_bvals))
        highb_mask = in_data > b0_threshold

        original_bvals = sorted(set(np.rint(in_data[highb_mask]).astype(int)))
        round_bvals = np.round(in_data, -2).astype(int)
        shell_bvals = sorted(set(round_bvals[highb_mask]))

        if len(shell_bvals) <= dsi_threshold:
            n_shells = len(shell_bvals)
            models = [n_shells]
            out_data = round_bvals.tolist()
            b_values = shell_bvals
        else:

            grid_search = GridSearchCV(
                KMeans(), param_grid={"n_clusters": range(1, 10)}, scoring=_rms
            ).fit(in_data[highb_mask].reshape(-1, 1))

            results = np.array(
                sorted(
                    zip(
                        grid_search.cv_results_["mean_test_score"] * -1.0,
                        grid_search.cv_results_["param_n_clusters"],
                    )
                )
            )

            models = results[:, 1].astype(int).tolist()
            n_shells = int(grid_search.best_params_["n_clusters"])

            out_data = np.zeros_like(in_data)
            predicted_shell = np.rint(
                np.squeeze(
                    grid_search.best_estimator_.cluster_centers_[
                        grid_search.best_estimator_.predict(
                            in_data[highb_mask].reshape(-1, 1)
                        )
                    ],
                )
            ).astype(int)

            if len(original_bvals) == n_shells:

                indices = np.abs(
                    predicted_shell[:, np.newaxis] - original_bvals
                ).argmin(axis=1)
                predicted_shell = original_bvals[indices]

            out_data[highb_mask] = predicted_shell
            out_data = np.round(out_data.astype(float), 2).tolist()
            b_values = sorted(
                np.unique(np.round(predicted_shell.astype(float), 2)).tolist()
            )

        b_masks = [(~highb_mask).tolist()] + [
            np.isclose(out_data, bvalue).tolist() for bvalue in b_values
        ]
        b_indices = [
            np.atleast_1d(np.squeeze(np.argwhere(b_mask)).astype(int)).tolist()
            for b_mask in b_masks
        ]

        b_dict = {
            int(round(k, 0)): value for k, value in zip([0] + b_values, b_indices)
        }

        return models, n_shells, out_data, b_values, b_masks, b_indices, b_dict


def _rms(estimator, X):
    """
    Callable to pass to GridSearchCV that will calculate a distance score.

    To consider: using `MDL
    <https://erikerlandson.github.io/blog/2016/08/03/x-medoids-using-minimum-description-length-to-identify-the-k-in-k-medoids/>`__

    """
    if len(np.unique(estimator.cluster_centers_)) < estimator.n_clusters:
        return -np.inf
    # Calculate distance from assigned shell centroid
    distance = X - estimator.cluster_centers_[estimator.predict(X)]
    # Make negative so CV optimizes minimizes the error
    return -np.sqrt(distance**2).sum()
