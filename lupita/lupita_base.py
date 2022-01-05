"""Main module."""
from sklearn.svm._base import _fit_liblinear
from sklearn.svm import _libsvm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel

"""
TODO in rough priority order (highest priority to lowest)

- Add init/fit/predict implementation
    - fit_libsvm

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
    
    @staticmethod
    def get_kernel_method(method: str, kernel_args):
        """
        TODO: More robust checks on valid arguments
        """
        if method == 'rbf':
            # Need to handle special case where gamma is chosen based off number of features
            # We create a small wrapper around rbf_kernel so that we can use the shape of X
            # to determine gamma.
            gamma = kernel_args['gamma']
            if gamma == 'auto':
                def rbf(X, Y):
                    n_samples, n_features = X.shape
                    gamma = 1.0 / n_features
                    return rbf_kernel(X, Y, gamma=gamma)
                kernel = rbf
            else:
                kernel = rbf_kernel

        if method == 'poly':
            degree = kernel_args['degree']
            kernel = functools.partial(poly_kernel, degree=degree)

        if method == 'linear':
            # TODO: Find how to use liblinear, since
            # it is optimized for linear SVM.
            kernel == linear_kernel

        return kernel
        
        
    
    def __init__(self,
                 C: float = 1,
                 gamma: float = 1,
                 kernel_x_method: str = 'rbf',
                 degree_x: int = 3,
                 gamma_x: float ='auto', # Float or str
                 kernel_xstar_method: str = 'rbf',
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
    
        if self.C < 0:
            raise ValueError("Penalty term must be positive; got (C=%r)" % self.C)

        self.C = C
        
        kernel_x_args = {
            'gamma': gamma_x,
            'degree': degree_x
        }
        
        kernel_star_args = {
           'gamma': gamma_xstar,
           'degree': degree_xstar
        }
        """
        What is this gamma and how is it different from gamma_x?
        """
        self.gamma = gamma
        self.kernel_x = LupitaSVC.get_kernel_method(kernel_x_method, kernel_x_args)
        self.kernel_xstar = LupitaSVC.get_kernel_method(kernel_xstar_method, kernel_xstar_args)
        

    def solve_for_label(self, X, y, X_star):
        return 0
    
    
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
        y = np.array(y).reshape(len(X), 1)
        X, y = check_X_y(X, y.flat, 'csr')
        XStar, y = check_X_y(XStar, y.flat, 'csr')
        
        # Consider using _validate_data as shown in SVC fit
        # Take caution with multi class
        check_classification_targets(y)
        """
        cls_labels = np.unique(y)
        """
        self.classes_ = np.unique(y)
        self.n_class = len(self.classes_)
        self.models = []
        
        # SVMPlus package added one to this as part of loop.
        # However that is just adding a constant value to kernel.
        # This was accompanied by a step, 'append bias'. But that would also
        # be incremented in each step of the loop
        K = kernel_method(X, X, kernel_param)
        KStar = kernel_method_star(XStar, XStar, kernel_param_star)
        # If that increment isn't needed, then we can compute this only once:
        G = np.eye(n_samples) - np.linalg.inv(np.eye(n_samples) + (self.C / self.gamma) * KStar)
        G = (1 / self.C) * G
        
        for c in range(0, n_class):
            # Train a model for each class, except for binary classifier
            y_temp = np.array(y) # Must reassign label for libsvm
            y[y != (c + 1)] = -1
            y[y == (c + 1)] =  1
            
            # TODO: Test other svm_type.
            # If we can use multiclass, then don't bother with this loop.
            model = svm._libsvm.fit(Q,
                                    np.ones(n_samples),
                                    svm_type=2, # One class SVM (but why?)
                                    nu = 1 / n_samples, # Sklearn default is 0.5
                                    kernel = 'precomputed',
                                    tol = self.tol,
                                    max_iter = self.max_iter)
            
            # model[0] is a list with indices of support vectors
            sv_x = X[model[0]] # X is training data
            sv_y = y[model[0]] # y is label

            coeff = model[3]
            dc = abs(coeff[0]).reshape(len(coeff[0]),1)
            dual_coef = np.multiply(sv_y, dc)
            support_vectors = sv_x  # support vector's features
            m = [support_vectors, dual_coef]

    
    # This is based on SKlearn SVC fit
    def old_fit(self, X, y, X_star=None):
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