"""
timeseries.py

Machine learning
with the minimum nescience principle

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
@web:       http://www.mathematicsunknown.com/
@copyright: GNU GPLv3
"""

#
# TODO List (open issues in github)
# - Support exogeneous parameters
# - Miscoding gvien exogeneous parameters
# - Support missing values
# - Support time series with dates (for example in predict)
#

import numpy  as np
import pandas as pd

import warnings

from sklearn.base             import BaseEstimator, RegressorMixin																					
from sklearn.utils.validation import check_is_fitted
from sklearn.utils            import column_or_1d
from sklearn.metrics          import mean_squared_error
from sklearn.utils            import check_array

import statsmodels.api as sm

from .surfeit    import Surfeit
from .inaccuracy import Inaccuracy
from .utils      import optimal_code_length

#
# State space representation of a time invariant Gaussian linear structural time series model
#
# Given the number of hidden states (k_states) the following matrices are fitted:
#  - Observations covariance H (scalar)
#  - Design matrix Z (k_states)
#  - Transition matrix T (k_states, k_states)
#  - State covariance diagonal matrix Q (k_states, k_states)
#
#  Selection matrix Eta(k_states, k_states) is fixed to the identity matrix
#
class _StateSpace(sm.tsa.statespace.MLEModel):

    start_params = None

    """
    Initialization of the class _StateSpace
        
    Parameters
    ----------
    endog:    array-like, shape (n_samples), the time series values.
    k_states: number of hidden states
    """            
    def __init__(self, endog, k_states):

        super(_StateSpace, self).__init__(endog, k_states=k_states, k_posdef=k_states)

        self.k_states = k_states

        # Observations + Design + Transition + State
        self.start_params = np.ones(1 + k_states + k_states*k_states + k_states)

        # Selection matrix R is the identity matrix
        for i in np.arange(k_states):
            self['selection', i, i] = 1.0

        # Initialize as approximate diffuse
        # and "burn" the firsts loglikelihood values
        self.initialize_approximate_diffuse()
        self.loglikelihood_burn = k_states

    """
    Update the params of the model.
    Function called internaly by MLEModel base class during optimization,
    for example, when the fit() method is called
    """            
    def update(self, params, *args, **kwargs):

        params = super(_StateSpace, self).update(params, *args, **kwargs)
        
        idx = 0

        # Observations covariance H
        self['obs_cov', 0] = params[idx]
        idx = idx + 1

        # Design matrix Z for an unidimensiona time series
        for i in np.arange(self.k_states):
            self['design', 0, i] = params[idx]
            idx = idx + 1

        # Transition matrix T
        for i in np.arange(self.k_states):
            for j in np.arange(self.k_states):
                self['transition', i, j] = params[idx]
                idx = idx + 1

        # State covariance matrix Q is a diagonal matrix
        for i in np.arange(self.k_states):
            self['state_cov', 0, i] = params[idx]
            idx = idx + 1


class TimeSeries(BaseEstimator, RegressorMixin):
    """
    Given a time series ts = {x1, ..., xn} composed by n samples, 
    computes a state space based model to forecast t future values of the series.

    Example of usage:
        
        from nescience.timeseries import TimeSeries

        ts = ...

        model = TimeSeries(auto=True)
        mode.fit(ts)
        model.predict(3)
    """


    def __init__(self, y_type="numeric", X_type=None, multivariate=False, auto=True, max_iter=100):
        """
        Initialization of the class TimeSeries
        
        Parameters
        ----------
        y_type:       The type of the time series, numeric or categorical
        X_type:       The type of the predictors, numeric, mixed or categorical,
                      in case of having a multivariate time series
                      None if the time series is univariate
        multivariate: "True" if we have other time series available as predictors
        auto:         "True" if we want to find automatically the optimal model
        max_iter:     maximum number of iterations allowed to fit parameters
        """        

        valid_y_types = ("numeric", "categorical")
        if y_type not in valid_y_types:
            raise ValueError("Valid options for 'y_type' are {}. "
                             "Got vartype={!r} instead."
                             .format(valid_y_types, y_type))

        if multivariate:
            valid_X_types = ("numeric", "mixed", "categorical")
            if X_type not in valid_X_types:
                raise ValueError("Valid options for 'X_type' are {}. "
                                 "Got vartype={!r} instead."
                                 .format(valid_X_types, X_type))

        self.y_type       = y_type
        self.X_type       = X_type
        self.multivariate = multivariate
        self.auto         = auto
        self.max_iter     = max_iter


    def fit(self, y, X=None):
        """
        Initialize the time series class with the actual data.
        If auto is set to True, train a model.
        
        Parameters
        ----------
        y : array-like, shape (n_samples)
            The time series.
        X : (optional) array-like, shape (n_samples, n_features)
            Time series features in case of a multivariate time series problem
            
        Returns
        -------
        self
        """

        self.y_ = column_or_1d(y)

        if self.y_type == "numeric":
            self.y_isnumeric = True
        else:
            self.y_isnumeric = False
        
        # Process X in case of a multivariate time series
        if self.multivariate:
            
            if X is None:
                raise ValueError("X argument is mandatory in case of multivariate time series.")

            if self.X_type == "mixed" or self.X_type == "categorical":

                if isinstance(X, pd.DataFrame):
                    self.X_isnumeric = [np.issubdtype(my_type, np.number) for my_type in X.dtypes]
                    self.X_ = np.array(X)
                else:
                    raise ValueError("Only DataFrame is allowed for X of type 'mixed' and 'categorical."
                                     "Got type {!r} instead."
                                     .format(type(X)))
                
            else:
                self.X_ = check_array(X)
                self.X_isnumeric = [True] * X.shape[1]

        # Auto Time Series
        if self.auto:

            self.surfeit_ = Surfeit()
            self.surfeit_.fit(y=self.y_)

            self.inaccuracy_ = Inaccuracy()
            self.inaccuracy_.fit(y=self.y_)

            self.model_ = self.StateSpace(self.max_iter)
                    
        return self


    def predict(self):
        """
        In-sample predictions    
                    
        Returns
        -------
        Array of the same size that the original time series
        with the predicted values.
        """
        
        check_is_fitted(self, "model_")
        
        predict = self.model_.predict()

        return predict


    def forecast(self, steps=1):
        """
        Out-of-sample forecasts    
        
        Parameters
        ----------
        steps : the number of steps to forecast from the end of the sample.
            
        Returns
        -------
        Array of size steps with the forecasted values.
        """
        
        check_is_fitted(self, "model_")
        
        forecast = self.model_.forecast(steps)

        return forecast


    def score(self):
        """
        Evaluate the performance of the current model.
        Compute the RMSE between time series and in-sample predictions

        Returns
        -------    
        Return root mean squared error
        """
        
        check_is_fitted(self, "model_")

        # Higher scores means better models
        mean = np.mean(self.y_)
        u = mean_squared_error(self.y_, self.model_.predict())
        v = np.mean([(self.y_[i] - mean)**2 for i in range(0, len(self.y_)-1)])
        score = 1 - u/v

        return score

        # rmse = -np.sqrt(mean_squared_error(self.y_, self.model_.predict()))
        # return rmse
		

    def get_model(self):
        """
        Get access to the private attribute model

        Returns
        -------		
        Return the fitted model
        """

        check_is_fitted(self, "model_")

        return self.model_

		
    def StateSpace(self, max_iter=100):
        """
        Learn empirically a state space model of k hidden states.
        
        Parameters
        ----------
        max_iter : maximum number of iterations allowed to fit parameters
            
        Returns
        -------
        (nescience, model, None)
        """ 

        #
        # Start search with a local level model
        #

        k_states  = 1
        ssm   = _StateSpace(self.y_, k_states=k_states)
        model = ssm.fit(disp=False, maxiter=max_iter)

        # Check if the model has failed to converge
        if not model.mle_retvals['converged']:
            raise ValueError("Model failed to converge. Please increase max_iter param.")

        nsc = self._nescience(model)

        #
        # Search for the best model
        #

        decreased = True
        while (decreased):
                        
            decreased = False
            k_states  = k_states + 1

            ssm       = _StateSpace(self.y_, k_states=k_states)
            new_model = ssm.fit(disp=False, maxiter=max_iter)

            # If the model failed to converge stop the search
            # but inform the user about the problem
            if not new_model.mle_retvals['converged']:
                warnings.warn("Search stopped because model failed to converge."
                              " Please consider increasing max_iter param.")
                break

            new_nsc = self._nescience(new_model)
        
            # Stop if nescience increases
            if new_nsc > nsc:
                break

            # Save data since nescience has decreased                        
            model     = new_model
            nsc       = new_nsc
        
        return model


    """
    Compute the nescience of a univariate time series based model
        
    Parameters
    ----------
    model : a trained time series model
                    
    Returns
    -------
    Return the nescience (float)
    """
    # TODO: Should use the generic class
    def _nescience(self, model):
        
        check_is_fitted(self)

        inaccuracy = self.inaccuracy_.inaccuracy_model(model)
        surfeit    = self.surfeit_.surfeit_model(model)

        # Avoid dividing by zero
        
        if surfeit == 0:
            surfeit = 10e-6
    
        if inaccuracy == 0:
            inaccuracy = 10e-6
            
        # TODO: Think about this problem
        if surfeit < inaccuracy:
            # The model is still too small to use surfeit
            surfeit = 1

        # Compute the nescience using an harmonic mean
        nescience = 2 / ( (1/inaccuracy) + (1/surfeit))
        
        return nescience       


    def auto_miscoding(self, min_lag=1, max_lag=None, mode='adjusted'):
        """
        Return the auto-miscoding of a time series, for a given number of lags

        Parameters
        ----------
        min_lag   : starting lag
        max_lag   : end lag. If none, the squared root of the number of samples is used
        mode      : the mode of miscoding, possible values are 'regular' for
                    the true miscoding, 'adjusted' for the normalized inverted
                    values, and 'partial' with positive and negative
                    contributions to dataset miscoding.
            
        Returns
        -------
        Return a numpy array with the lagged miscodings
        """

        check_is_fitted(self)

        valid_modes = ('regular', 'adjusted', 'partial')

        if mode not in valid_modes:
            raise ValueError("Valid options for 'mode' are {}. "
                             "Got mode={!r} instead."
                            .format(valid_modes, mode))
        
        lag_mscd = list()
        
        # Use a default value for max_lag
        if max_lag == None:
            max_lag = int(np.sqrt(self.y_.shape[0]))

        for i in np.arange(start=min_lag, stop=max_lag):

            # Compute lagged vectors
            new_y = self.y_.copy()
            new_y = np.roll(new_y, -i)
            new_y = new_y[:-i]
            new_x = self.y_.copy()
            new_x = new_x[:-i]

            # Compute miscoding
            ldm_y  = optimal_code_length(x1=new_y, numeric1=self.y_isnumeric)
            ldm_X  = optimal_code_length(x1=new_x, numeric1=self.y_isnumeric)
            ldm_Xy = optimal_code_length(x1=new_x, numeric1=self.y_isnumeric, x2=new_y, numeric2=self.y_isnumeric)
            mscd   = ( ldm_Xy - min(ldm_X, ldm_y) ) / max(ldm_X, ldm_y)
                
            lag_mscd.append(mscd)
                
        regular = np.array(lag_mscd)

        if mode == 'regular':
            return regular

        elif mode == 'adjusted':

            adjusted = 1 - regular
            if np.sum(adjusted) != 0:
                adjusted = adjusted / np.sum(adjusted)
            return adjusted

        elif mode == 'partial':

            adjusted = 1 - regular
            if np.sum(adjusted) != 0:
                adjusted = adjusted / np.sum(adjusted)

            if np.sum(regular) != 0:
                partial  = adjusted - regular / np.sum(regular)
            else:
                partial  = adjusted
            return partial

        else:
            raise ValueError("Valid options for 'mode' are {}. "
                             "Got mode={!r} instead."
                            .format(valid_modes, mode))


    def cross_miscoding(self, attribute, min_lag=1, max_lag=None, mode='adjusted'):
        """
        Return the cross-miscoding of the target time series and a second time series
        in case of multivariate time series

        Parameters
        ----------
        attribute : the attribute of the second series
        min_lag   : starting lag
        max_lag   : end lag. If none, the squared root of the number of samples is used.
        mode      : the mode of miscoding, possible values are 'regular' for
                    the true miscoding, 'adjusted' for the normalized inverted
                    values, and 'partial' with positive and negative
                    contributions to dataset miscoding.
            
        Returns
        -------
        Return a numpy array with the lagged miscodings
        """        

        check_is_fitted(self)

        valid_modes = ('regular', 'adjusted', 'partial')

        if mode not in valid_modes:
            raise ValueError("Valid options for 'mode' are {}. "
                             "Got mode={!r} instead."
                            .format(valid_modes, mode))
        
        lag_mscd = list()
        
        # Use a default value for max_lag
        if max_lag == None:
            max_lag = int(np.sqrt(self.X_.shape[0]))

        for i in np.arange(start=min_lag, stop=max_lag):

            # Compute lagged vectors
            new_y = self.y_.copy()
            new_y = np.roll(new_y, -i)
            new_y = new_y[:-i]
            new_x = self.X_[:,attribute].copy()
            new_x = new_x[:-i]

            # Compute miscoding
            ldm_y  = optimal_code_length(x1=new_y, numeric1=self.y_isnumeric)
            ldm_X  = optimal_code_length(x1=new_x, numeric1=self.y_isnumeric)
            ldm_Xy = optimal_code_length(x1=new_x, numeric1=self.y_isnumeric, x2=new_y, numeric2=self.y_isnumeric)
            mscd   = ( ldm_Xy - min(ldm_X, ldm_y) ) / max(ldm_X, ldm_y)
                
            lag_mscd.append(mscd)
                
        regular = np.array(lag_mscd)

        if mode == 'regular':
            return regular

        elif mode == 'adjusted':

            adjusted = 1 - regular
            if np.sum(adjusted) != 0:
                adjusted = adjusted / np.sum(adjusted)
            return adjusted

        elif mode == 'partial':

            adjusted = 1 - regular
            if np.sum(adjusted) != 0:
                adjusted = adjusted / np.sum(adjusted)

            if np.sum(regular) != 0:
                partial  = adjusted - regular / np.sum(regular)
            else:
                partial  = adjusted
            return partial

        else:
            raise ValueError("Valid options for 'mode' are {}. "
                             "Got mode={!r} instead."
                            .format(valid_modes, mode))

# from datetime import datetime

# passengers = pd.read_stata("../../StateSpace/air2.dta")
# passengers.index = pd.date_range(start=datetime(passengers.time[0], 1, 1), periods=len(passengers), freq='MS')

# data  = passengers["air"]

# mod = TimeSeries(auto=True)
# mod.fit(data)