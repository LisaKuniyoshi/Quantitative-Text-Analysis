import numpy as np
import warnings

from statsmodels.tools.validation import bool_like
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.tools import nan_dot
from statsmodels.stats.contrast import ContrastResults
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.discrete.discrete_margins import DiscreteMargins

# statsmodels.base.model
class DiscreteMarginsResults(DiscreteMargins, LikelihoodModelResults):
    # TODO: conf_int override issue
    # Base classes for class "DiscreteMarginsResults" define method "conf_int" in incompatible way
    #   Positional parameter count mismatch; base method has 3, but override has 2
    #   Parameter 3 mismatch: base parameter "cols" is keyword parameter, override parameter is position-onlyPylancereportIncompatibleMethodOverride
    # model.py(2160, 9): Base class "LikelihoodModelResults" provides type "(self: Self@LikelihoodModelResults, alpha: float = 0.05, cols: Unknown | None = None) -> NDArray[Any]", which is overridden
    # discrete_margins.py(497, 9): Base class "DiscreteMargins" overrides with type "(self: Self@DiscreteMargins, alpha: float = 0.05) -> NDArray[Any]"    
    def __init__(self, results, args, params, kwargs=None):
        DiscreteMargins.__init__(self, results, args, kwargs)
        LikelihoodModelResults.__init__(self, results.model, params)

        if "use_t" in kwargs:
            use_t = kwargs["use_t"]
            self.use_t = use_t if use_t is not None else False

        # TODO: kwargsの衝突。DiscreteMarginsの__init__に渡すものと、LikelihoodModelResultsライクなinitで使うものが混在

    def normalized_cov_params(self):
        """See specific model class docstring"""
        raise NotImplementedError

    def wald_test(
        self,
        r_matrix,
        cov_p=None,
        invcov=None,
        use_f=None,
        df_constraints=None,
        scalar=None,
    ):
        """
        Compute a Wald-test for a joint linear hypothesis.

        Parameters
        ----------
        r_matrix : {array_like, str, tuple}
            One of:

            - array : An r x k array where r is the number of restrictions to
              test and k is the number of regressors. It is assumed that the
              linear combination is equal to zero.
            - str : The full hypotheses to test can be given as a string.
              See the examples.
            - tuple : A tuple of arrays in the form (R, q), ``q`` can be
              either a scalar or a length p row vector.

        cov_p : array_like, optional
            An alternative estimate for the parameter covariance matrix.
            If None is given, self.normalized_cov_params is used.
        invcov : array_like, optional
            A q x q array to specify an inverse covariance matrix based on a
            restrictions matrix.
        use_f : bool
            If True, then the F-distribution is used. If False, then the
            asymptotic distribution, chisquare is used. If use_f is None, then
            the F distribution is used if the model specifies that use_t is True.
            The test statistic is proportionally adjusted for the distribution
            by the number of constraints in the hypothesis.
        df_constraints : int, optional
            The number of constraints. If not provided the number of
            constraints is determined from r_matrix.
        scalar : bool, optional
            Flag indicating whether the Wald test statistic should be returned
            as a sclar float. The current behavior is to return an array.
            This will switch to a scalar float after 0.14 is released. To
            get the future behavior now, set scalar to True. To silence
            the warning and retain the legacy behavior, set scalar to
            False.

        Returns
        -------
        ContrastResults
            The results for the test are attributes of this results instance.

        See Also
        --------
        f_test : Perform an F tests on model parameters.
        t_test : Perform a single hypothesis test.
        statsmodels.stats.contrast.ContrastResults : Test results.
        patsy.DesignInfo.linear_constraint : Specify a linear constraint.

        Notes
        -----
        The matrix `r_matrix` is assumed to be non-singular. More precisely,

        r_matrix (pX pX.T) r_matrix.T

        is assumed invertible. Here, pX is the generalized inverse of the
        design matrix of the model. There can be problems in non-OLS models
        where the rank of the covariance of the noise is not full.
        """
        use_f = bool_like(use_f, "use_f", strict=True, optional=True)
        scalar = bool_like(scalar, "scalar", strict=True, optional=True)
        if use_f is None:
            # switch to use_t false if undefined
            use_f = hasattr(self, "use_t") and self.use_t

        if self.params.ndim == 2:
            names = [f"y{i[0]}_{i[1]}" for i in self.model.data.cov_names]
        else:
            names = self.model.data.cov_names
        params = self.params.ravel(order="F")

        # TODO: 本来は mgr = statsmodels.formula._manager.FormulaManager()だが、外部からは使えないので、代わりに以下の代替機能を用意する
        # get_linear_constraints(r_matrix, names)
        lc = mgr.get_linear_constraints(r_matrix, names)
        r_matrix, q_matrix = lc.constraint_matrix, lc.constraint_values

        if (
            self.normalized_cov_params is None
            and cov_p is None
            and invcov is None
            and not hasattr(self, "cov_params_default")
        ):
            raise ValueError(
                "need covariance of parameters for computing F statistics"
            )

        cparams = np.dot(r_matrix, params[:, None])
        J = float(r_matrix.shape[0])  # number of restrictions

        if q_matrix is None:
            q_matrix = np.zeros(J)
        else:
            q_matrix = np.asarray(q_matrix)
        if q_matrix.ndim == 1:
            q_matrix = q_matrix[:, None]
            if q_matrix.shape[0] != J:
                raise ValueError(
                    "r_matrix and q_matrix must have the same number of rows"
                )
        Rbq = cparams - q_matrix
        if invcov is None:
            cov_p = self.cov_params(r_matrix=r_matrix, cov_p=cov_p)
            if np.isnan(cov_p).max():
                raise ValueError(
                    "r_matrix performs f_test for using "
                    "dimensions that are asymptotically "
                    "non-normal"
                )
            invcov = np.linalg.pinv(cov_p)
            J_ = np.linalg.matrix_rank(cov_p)
            if J_ < J:
                warnings.warn(
                    "covariance of constraints does not have full "
                    "rank. The number of constraints is %d, but "
                    "rank is %d" % (J, J_),
                    ValueWarning,
                    stacklevel=2,
                )
                J = J_

        if df_constraints is not None:
            # let caller override J by df_constraint
            J = df_constraints

        if hasattr(self, "mle_settings") and self.mle_settings["optimizer"] in [
            "l1",
            "l1_cvxopt_cp",
        ]:
            F = nan_dot(nan_dot(Rbq.T, invcov), Rbq)
        else:
            F = np.dot(np.dot(Rbq.T, invcov), Rbq)

        df_resid = getattr(self, "df_resid_inference", self.df_resid)
        if scalar is None:
            warnings.warn(
                "The behavior of wald_test will change after 0.14 to returning "
                "scalar test statistic values. To get the future behavior now, "
                "set scalar to True. To silence this message while retaining "
                "the legacy behavior, set scalar to False.",
                FutureWarning,
                stacklevel=2,
            )
            scalar = False
        if scalar and F.size == 1:
            F = float(np.squeeze(F))
        if use_f:
            F /= J
            return ContrastResults(F=F, df_denom=df_resid, df_num=J)  # invcov.shape[0])
        else:
            return ContrastResults(
                chi2=F, df_denom=J, statistic=F, distribution="chi2", distargs=(J,)
            )
