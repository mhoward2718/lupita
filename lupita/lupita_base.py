"""Main module."""
from sklearn.svm._base import _fit_liblinear, # Also import LibSVM fit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel

"""
TODO in rough priority order (highest priority to lowest)

- Add good docstrings
    + To do this, create a PyCharm project instead of a Jupyter Notebook to edit.

- Create and populate an 'Examples' directory.
    + Illustrate the effectiveness of learning with privleged information by demonstrating improved test performance versus model without PI.
        - Some possible examples:
            + Digit recogition with wavelet transform as PI
            + Digit recognition of low resolution images with high-res as PI (as SVMPlus did)
            + Choose some UCI classification data with randomly chosen features assigned to PI space
    + These notebooks should add commentary expaining...
        + What LUPI is
        + How it changes the formulation of SVM
        + How it changes model performance
        + How it the fit model is different (i.e., coefficients in the case of a linear kernel)
        + Visualization of decision
        + Etc

- Determine if we can still use SKLearn Pipeline class with a LupitaSVC instance. 
    + Specifically, does the additional parameter, XStar, cause a problem when the pipeline calls .fit on the model?
    
- Use LibLinear when (both) kernel is linear/none instead of calling libsvm with no kernel.
    + But what's the difference when we have to precompute the kernel?

- Support solving with libraries POGS, LASVM, ThunderSVM, LIBIRWLS, etc,
    + First, need to determine:
        - Which are best speed and scale wise?
        - Which can be used with PI?



"""

class LupitaSVC(BaseEstimator, ClassifierMixin):
    """SVM classifier training with privileged information.
   
    TODO: Add references to original LUPI paper
    
    """
    
    def get_kernel_method(method: str):
        kernels = {
            'linear': linear_kernel,
            'rbf': rbf_kernel,
            'poly': polynomial_kernel
        }
        from functools import partial
        kernel = partial(kernels[method])
        return kernel
        
        
    
    def __init__(self,
                 C: float = 1,
                 gamma: float = 1,
                 kernel_x: str = 'rbf',
                 degree_x: int = 3,
                 gamma_x: float ='auto', # Float or str
                 kernel_xstar: str = 'rbf',
                 degree_xstar: int = 3,
                 gamma_xstar: float = 'auto',
                 max_iter: int = 100000,
                 tol: float = 1e-5):
        """
        TODO: Use Pycharm to generate better docstrings for all methods
        """
        if gamma == 0 or gamma_x == 0:
            msg = ("The value gamma or gamma_x is 0.0, which is invalid. Consider passing 'auto' to set 1 / n_features.")
            raise ValueError(msg)

        self.C = C
        self.gamma = gamma
        self.degree_x = degree_x
        self.gamma_x = gamma_x
        self.degree_xstar = degree_xstar
        self.gamma_xstar = gamma_xstar
        self.kernel_x = get_kernel_method(kernel_x)
        self.kernel_xstar = get_kernel_method(kernel_xstar)
        

    def solve_for_label(self, X, y, X_star):
        return 0
    
    # This is based on SKlearn SVC fit
    def fit(self, X, y, X_star=None):
        """Fit the model according to the given training data (X) and privileged information (X_star)
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        X_star : array-like of shape (n_samples, n_features), default=None
             Privileged information vector, where `n_samples` is the number of samples and
            `n_features` is the number of features in the privileged information space.

        Returns
        -------
        self : object
            An instance of the estimator.
        """
        if self.C < 0:
            raise ValueError("Penalty term must be positive; got (C=%r)" % self.C)

        
        X, y = self._validate_data(
            X,
            y,
            accept_sparse="csr",
            dtype=np.float64,
            order="C",
            accept_large_sparse=False,
        )
        check_classification_targets(y)
        self.classes_ = np.unique(y)

        self.coef_, self.intercept_, self.n_iter_ = _fit_liblinear(
            X,
            y,
            self.C,
            self.fit_intercept,
            self.intercept_scaling,
            self.class_weight,
            self.penalty,
            self.dual,
            self.verbose,
            self.max_iter,
            self.tol,
            self.random_state,
            self.multi_class,
            self.loss,
            sample_weight=sample_weight,
        )

        if self.multi_class == "crammer_singer" and len(self.classes_) == 2:
            self.coef_ = (self.coef_[1] - self.coef_[0]).reshape(1, -1)
            if self.fit_intercept:
                intercept = self.intercept_[1] - self.intercept_[0]
                self.intercept_ = np.array([intercept])

        return self