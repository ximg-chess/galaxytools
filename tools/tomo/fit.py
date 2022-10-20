#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 15:36:22 2021

@author: rv43
"""

import sys
import re
import logging
import numpy as np

from asteval import Interpreter
from copy import deepcopy
#from lmfit import Minimizer
from lmfit import Model, Parameters
from lmfit.models import ConstantModel, LinearModel, QuadraticModel, PolynomialModel,\
        StepModel, RectangleModel, GaussianModel, LorentzianModel

from general import is_index, index_nearest, quickPlot

# sigma = fwhm_factor*fwhm
fwhm_factor = {
    'gaussian' : f'fwhm/(2*sqrt(2*log(2)))',
    'lorentzian' : f'0.5*fwhm',
    'splitlorentzian' : f'0.5*fwhm', # sigma = sigma_r
    'voight' : f'0.2776*fwhm', # sigma = gamma
    'pseudovoight' : f'0.5*fwhm'} # fraction = 0.5

# amplitude = height_factor*height*fwhm
height_factor = {
    'gaussian' : f'height*fwhm*0.5*sqrt(pi/log(2))',
    'lorentzian' : f'height*fwhm*0.5*pi',
    'splitlorentzian' : f'height*fwhm*0.5*pi', # sigma = sigma_r
    'voight' : f'3.334*height*fwhm', # sigma = gamma
    'pseudovoight' : f'1.268*height*fwhm'} # fraction = 0.5

class Fit:
    """Wrapper class for lmfit
    """
    def __init__(self, x, y, models=None, **kwargs):
        self._x = x
        self._y = y
        self._model = None
        self._parameters = Parameters()
        self._result = None
        if models is not None:
            if callable(models) or isinstance(models, str):
                kwargs = self.add_model(models, **kwargs)
            elif isinstance(models, (tuple, list)):
                for model in models:
                    kwargs = self.add_model(model, **kwargs)
            self.fit(**kwargs)

    @classmethod
    def fit_data(cls, x, y, models, **kwargs):
        return cls(x, y, models, **kwargs)

    @property
    def best_errors(self):
        if self._result is None:
            return None
        errors = {}
        names = sorted(self._result.params)
        for name in names:
            par = self._result.params[name]
            errors[name] = par.stderr
        return errors

    @property
    def best_fit(self):
        if self._result is None:
            return None
        return self._result.best_fit

    @property
    def best_parameters(self):
        if self._result is None:
            return None
        parameters = []
        names = sorted(self._result.params)
        for name in names:
            par = self._result.params[name]
            parameters.append({'name' : par.name, 'value' : par.value, 'error' : par.stderr,
                    'init_value' : par.init_value, 'min' : par.min, 'max' : par.max,
                    'vary' : par.vary, 'expr' : par.expr})
        return parameters

    @property
    def best_values(self):
        if self._result is None:
            return None
        return self._result.params.valuesdict()

    @property
    def chisqr(self):
        return self._result.chisqr

    @property
    def covar(self):
        return self._result.covar

    @property
    def init_values(self):
        if self._result is None:
            return None
        return self._result.init_values

    @property
    def num_func_eval(self):
        return self._result.nfev

    @property
    def redchi(self):
        return self._result.redchi

    @property
    def residual(self):
        return self._result.residual

    @property
    def success(self):
        if not self._result.success:
#            print(f'ier = {self._result.ier}')
#            print(f'lmdif_message = {self._result.lmdif_message}')
#            print(f'message = {self._result.message}')
#            print(f'nfev = {self._result.nfev}')
#            print(f'redchi = {self._result.redchi}')
#            print(f'success = {self._result.success}')
            if self._result.ier == 0 or self._result.ier == 5:
                logging.warning(f'ier = {self._result.ier}: {self._result.message}')
            else:
                logging.warning(f'ier = {self._result.ier}: {self._result.message}')
                return True
#            self.print_fit_report()
#            self.plot()
        return self._result.success

    @property
    def var_names(self):
        """Intended to be used with covar
        """
        if self._result is None:
            return None
        return self._result.var_names

    def print_fit_report(self, show_correl=False):
        if self._result is not None:
            print(self._result.fit_report(show_correl=show_correl))

    def add_parameter(self, **parameter):
        if not isinstance(parameter, dict):
            illegal_value(parameter, 'parameter', 'add_parameter')
            return
        self._parameters.add(**parameter)

    def add_model(self, model, prefix=None, parameters=None, **kwargs):
        # Create the new model
#        print('\nAt start adding model:')
#        self._parameters.pretty_print()
        if prefix is not None and not isinstance(prefix, str):
            logging.warning('Ignoring illegal prefix: {model} {type(model)}')
            prefix = None
        if callable(model):
            newmodel = Model(model, prefix=prefix)
        elif isinstance(model, str):
            if model == 'constant':
                newmodel = ConstantModel(prefix=prefix)
            elif model == 'linear':
                newmodel = LinearModel(prefix=prefix)
            elif model == 'quadratic':
                newmodel = QuadraticModel(prefix=prefix)
            elif model == 'gaussian':
                newmodel = GaussianModel(prefix=prefix)
            elif model == 'step':
                form = kwargs.get('form')
                if form is not None:
                    del kwargs['form']
                if form is None or form not in ('linear', 'atan', 'arctan', 'erf', 'logistic'):
                    logging.error(f'Illegal form parameter for build-in step model ({form})')
                    return kwargs
                newmodel = StepModel(prefix=prefix, form=form)
            elif model == 'rectangle':
                form = kwargs.get('form')
                if form is not None:
                    del kwargs['form']
                if form is None or form not in ('linear', 'atan', 'arctan', 'erf', 'logistic'):
                    logging.error(f'Illegal form parameter for build-in rectangle model ({form})')
                    return kwargs
                newmodel = RectangleModel(prefix=prefix, form=form)
            else:
                logging.error('Unknown build-in fit model')
                return kwargs
        else:
            illegal_value(model, 'model', 'add_model')
            return kwargs

        # Add the new model to the current one
        if self._model is None:
            self._model = newmodel
        else:
            self._model += newmodel
        if self._parameters is None:
            self._parameters = newmodel.make_params()
        else:
            self._parameters += newmodel.make_params()
#        print('\nAfter adding model:')
#        self._parameters.pretty_print()

        # Initialize the model parameters
        if prefix is None:
            prefix = ""
        if parameters is not None:
            if not isinstance(parameters, (tuple, list)):
                illegal_value(parameters, 'parameters', 'add_model')
                return kwargs
            for parameter in parameters:
                if not isinstance(parameter, dict):
                    illegal_value(parameter, 'parameter in parameters', 'add_model')
                    return kwargs
                parameter['name']  = prefix+parameter['name']
                self._parameters.add(**parameter)
        for name, value in kwargs.items():
            if isinstance(value, (int, float)):
                self._parameters.add(prefix+name, value=value)
#        print('\nAt end add_model:')
#        self._parameters.pretty_print()

        return kwargs

    def fit(self, interactive=False, guess=False, **kwargs):
        if self._model is None:
            logging.error('Undefined fit model')
            return
        # Current parameter values
        pars = self._parameters.valuesdict()
        # Apply parameter updates through keyword arguments
        for par in set(pars) & set(kwargs):
            pars[par] = kwargs.pop(par)
            self._parameters[par].set(value=pars[par])
        # Check for uninitialized parameters
        for par, value in pars.items():
            if value is None or np.isinf(value) or np.isnan(value):
                if interactive:
                    self._parameters[par].set(value=
                            input_num(f'Enter an initial value for {par}: '))
                else:
                    self._parameters[par].set(value=1.0)
#        print('\nAt start actual fit:')
#        print(f'kwargs = {kwargs}')
#        self._parameters.pretty_print()
#        print(f'parameters:\n{self._parameters}')
#        print(f'x = {self._x}')
#        print(f'len(x) = {len(self._x)}')
#        print(f'y = {self._y}')
#        print(f'len(y) = {len(self._y)}')
        if guess:
            self._parameters = self._model.guess(self._y, x=self._x)
        self._result = self._model.fit(self._y, self._parameters, x=self._x, **kwargs)
#        print('\nAt end actual fit:')
#        print(f'var_names:\n{self._result.var_names}')
#        print(f'stderr:\n{np.sqrt(np.diagonal(self._result.covar))}')
#        self._parameters.pretty_print()
#        print(f'parameters:\n{self._parameters}')

    def plot(self):
        if self._result is None:
            return
        components = self._result.eval_components()
        plots = ((self._x, self._y, '.'), (self._x, self._result.best_fit, 'k-'),
                (self._x, self._result.init_fit, 'g-'))
        legend = ['data', 'best fit', 'init']
        if len(components) > 1:
            for modelname, y in components.items():
                if isinstance(y, (int, float)):
                    y *= np.ones(len(self._x))
                plots += ((self._x, y, '--'),)
#                if modelname[-1] == '_':
#                    legend.append(modelname[:-1])
#                else:
#                    legend.append(modelname)
        quickPlot(plots, legend=legend, block=True)

    @staticmethod
    def guess_init_peak(x, y, *args, center_guess=None, use_max_for_center=True):
        """ Return a guess for the initial height, center and fwhm for a peak
        """
        center_guesses = None
        if len(x) != len(y):
            logging.error(f'Illegal x and y lengths ({len(x)}, {len(y)}), skip initial guess')
            return None, None, None
        if isinstance(center_guess, (int, float)):
            if len(args):
                logging.warning('Ignoring additional arguments for single center_guess value')
        elif isinstance(center_guess, (tuple, list, np.ndarray)):
            if len(center_guess) == 1:
                logging.warning('Ignoring additional arguments for single center_guess value')
                if not isinstance(center_guess[0], (int, float)):
                    raise ValueError(f'Illegal center_guess type ({type(center_guess[0])})')
                center_guess = center_guess[0]
            else:
                if len(args) != 1:
                    raise ValueError(f'Illegal number of arguments ({len(args)})')
                n = args[0]
                if not is_index(n, 0, len(center_guess)):
                    raise ValueError('Illegal argument')
                center_guesses = center_guess
                center_guess = center_guesses[n]
        elif center_guess is not None:
            raise ValueError(f'Illegal center_guess type ({type(center_guess)})')

        # Sort the inputs
        index = np.argsort(x)
        x = x[index]
        y = y[index]
        miny = y.min()
#        print(f'miny = {miny}')
#        print(f'x_range = {x[0]} {x[-1]} {len(x)}')
#        print(f'y_range = {y[0]} {y[-1]} {len(y)}')

#        xx = x
#        yy = y
        # Set range for current peak
#        print(f'center_guesses = {center_guesses}')
        if center_guesses is not None:
            if n == 0:
               low = 0
               upp = index_nearest(x, (center_guesses[0]+center_guesses[1])/2)
            elif n == len(center_guesses)-1:
               low = index_nearest(x, (center_guesses[n-1]+center_guesses[n])/2)
               upp = len(x)
            else:
               low = index_nearest(x, (center_guesses[n-1]+center_guesses[n])/2)
               upp = index_nearest(x, (center_guesses[n]+center_guesses[n+1])/2)
#            print(f'low = {low}')
#            print(f'upp = {upp}')
            x = x[low:upp]
            y = y[low:upp]
#            quickPlot(x, y, vlines=(x[0], center_guess, x[-1]), block=True)

        # Estimate FHHM
        maxy = y.max()
#        print(f'x_range = {x[0]} {x[-1]} {len(x)}')
#        print(f'y_range = {y[0]} {y[-1]} {len(y)} {miny} {maxy}')
#        print(f'center_guess = {center_guess}')
        if center_guess is None:
            center_index = np.argmax(y)
            center = x[center_index]
            height = maxy-miny
        else:
            if use_max_for_center:
                center_index = np.argmax(y)
                center = x[center_index]
                if center_index < 0.1*len(x) or center_index > 0.9*len(x):
                    center_index = index_nearest(x, center_guess)
                    center = center_guess
            else:
                center_index = index_nearest(x, center_guess)
                center = center_guess
            height = y[center_index]-miny
#        print(f'center_index = {center_index}')
#        print(f'center = {center}')
#        print(f'height = {height}')
        half_height = miny+0.5*height
#        print(f'half_height = {half_height}')
        fwhm_index1 = 0
        for i in range(center_index, fwhm_index1, -1):
            if y[i] < half_height:
                fwhm_index1 = i
                break
#        print(f'fwhm_index1 = {fwhm_index1} {x[fwhm_index1]}')
        fwhm_index2 = len(x)-1
        for i in range(center_index, fwhm_index2):
            if y[i] < half_height:
                fwhm_index2 = i
                break
#        print(f'fwhm_index2 = {fwhm_index2} {x[fwhm_index2]}')
#        quickPlot((x,y,'o'), vlines=(x[fwhm_index1], center, x[fwhm_index2]), block=True)
        if fwhm_index1 == 0 and fwhm_index2 < len(x)-1:
            fwhm = 2*(x[fwhm_index2]-center)
        elif fwhm_index1 > 0 and fwhm_index2 == len(x)-1:
            fwhm = 2*(center-x[fwhm_index1])
        else:
            fwhm = x[fwhm_index2]-x[fwhm_index1]
#        print(f'fwhm_index1 = {fwhm_index1} {x[fwhm_index1]}')
#        print(f'fwhm_index2 = {fwhm_index2} {x[fwhm_index2]}')
#        print(f'fwhm = {fwhm}')

        # Return height, center and FWHM
#        quickPlot((x,y,'o'), (xx,yy), vlines=(x[fwhm_index1], center, x[fwhm_index2]), block=True)
        return height, center, fwhm


class FitMultipeak(Fit):
    """Fit data with multiple peaks
    """
    def __init__(self, x, y, normalize=True):
        super().__init__(x, deepcopy(y))
        self._norm = None
        self._fwhm_max = None
        self._sigma_max = None
        if normalize:
            self._normalize()
        #quickPlot((self._x,self._y), block=True)

    @classmethod
    def fit_multipeak(cls, x, y, centers, peak_models='gaussian', center_exprs=None, fit_type=None,
            background_order=None, fwhm_max=None, plot_components=None):
        """Make sure that centers and fwhm_max are in the correct units and consistent with expr
           for a uniform fit (fit_type == 'uniform')
        """
        fit = cls(x, y)
        success = fit.fit(centers, fit_type=fit_type, peak_models=peak_models, fwhm_max=fwhm_max,
                center_exprs=center_exprs, background_order=background_order,
                plot_components=plot_components)
        if success:
            return fit.best_fit, fit.residual, fit.best_values, fit.best_errors, fit.redchi, \
                    fit.success
        else:
            return np.array([]), np.array([]), {}, {}, sys.float_info.max, False

    def fit(self, centers, fit_type=None, peak_models=None, center_exprs=None, fwhm_max=None,
                background_order=None, plot_components=None, param_constraint=False):
        self._fwhm_max = fwhm_max
        # Create the multipeak model
        self._create_model(centers, fit_type, peak_models, center_exprs, background_order,
                param_constraint)

        # Perform the fit
        try:
            if param_constraint:
                super().fit(fit_kws={'xtol' : 1.e-5, 'ftol' : 1.e-5, 'gtol' : 1.e-5})
            else:
                super().fit()
        except:
            return False

        # Check for valid fit parameter results
        fit_failure = self._check_validity()
        success = True
        if fit_failure:
            if param_constraint:
                logging.warning('  -> Should not happen with param_constraint set, fail the fit')
                success = False
            else:
                logging.info('  -> Retry fitting with constraints')
                self.fit(centers, fit_type, peak_models, center_exprs, fwhm_max=fwhm_max,
                        background_order=background_order, plot_components=plot_components,
                        param_constraint=True)
        else:
            # Renormalize the data and results
            self._renormalize()

            # Print report and plot components if requested
            if plot_components is not None:
                self.print_fit_report()
                self.plot()

        return success

    def _create_model(self, centers, fit_type=None, peak_models=None, center_exprs=None,
                background_order=None, param_constraint=False):
        """Create the multipeak model
        """
        if isinstance(centers, (int, float)):
            centers = [centers]
        num_peaks = len(centers)
        if peak_models is None:
            peak_models = num_peaks*['gaussian']
        elif isinstance(peak_models, str):
            peak_models = num_peaks*[peak_models]
        if len(peak_models) != num_peaks:
            raise ValueError(f'Inconsistent number of peaks in peak_models ({len(peak_models)} vs '+
                    f'{num_peaks})')
        if num_peaks == 1:
            if fit_type is not None:
                logging.warning('Ignoring fit_type input for fitting one peak')
            fit_type = None
            if center_exprs is not None:
                logging.warning('Ignoring center_exprs input for fitting one peak')
                center_exprs = None
        else:
            if fit_type == 'uniform':
                if center_exprs is None:
                    center_exprs = [f'scale_factor*{cen}' for cen in centers]
                if len(center_exprs) != num_peaks:
                    raise ValueError(f'Inconsistent number of peaks in center_exprs '+
                            f'({len(center_exprs)} vs {num_peaks})')
            elif fit_type == 'unconstrained' or fit_type is None:
                if center_exprs is not None:
                    logging.warning('Ignoring center_exprs input for unconstrained fit')
                    center_exprs = None
            else:
                raise ValueError(f'Illegal fit_type in fit_multigaussian {fit_type}')
        self._sigma_max = None
        if param_constraint:
            min_value = sys.float_info.min
            if self._fwhm_max is not None:
                self._sigma_max = np.zeros(num_peaks)
        else:
            min_value = None

        # Reset the fit
        self._model = None
        self._parameters = Parameters()
        self._result = None

        # Add background model
        if background_order is not None:
            if background_order == 0:
                self.add_model('constant', prefix='background', c=0.0)
            elif background_order == 1:
                self.add_model('linear', prefix='background', slope=0.0, intercept=0.0)
            elif background_order == 2:
                self.add_model('quadratic', prefix='background', a=0.0, b=0.0, c=0.0)
            else:
                raise ValueError(f'background_order = {background_order}')

        # Add peaks and guess initial fit parameters
        ast = Interpreter()
        if num_peaks == 1:
            height_init, cen_init, fwhm_init = self.guess_init_peak(self._x, self._y)
            if self._fwhm_max is not None and fwhm_init > self._fwhm_max:
                fwhm_init = self._fwhm_max
            ast(f'fwhm = {fwhm_init}')
            ast(f'height = {height_init}')
            sig_init = ast(fwhm_factor[peak_models[0]])
            amp_init = ast(height_factor[peak_models[0]])
            sig_max = None
            if self._sigma_max is not None:
                ast(f'fwhm = {self._fwhm_max}')
                sig_max = ast(fwhm_factor[peak_models[0]])
                self._sigma_max[0] = sig_max
            self.add_model(peak_models[0], parameters=(
                    {'name' : 'amplitude', 'value' : amp_init, 'min' : min_value},
                    {'name' : 'center', 'value' : cen_init, 'min' : min_value},
                    {'name' : 'sigma', 'value' : sig_init, 'min' : min_value, 'max' : sig_max}))
        else:
            if fit_type == 'uniform':
                self.add_parameter(name='scale_factor', value=1.0)
            for i in range(num_peaks):
                height_init, cen_init, fwhm_init = self.guess_init_peak(self._x, self._y, i,
                        center_guess=centers)
                if self._fwhm_max is not None and fwhm_init > self._fwhm_max:
                    fwhm_init = self._fwhm_max
                ast(f'fwhm = {fwhm_init}')
                ast(f'height = {height_init}')
                sig_init = ast(fwhm_factor[peak_models[i]])
                amp_init = ast(height_factor[peak_models[i]])
                sig_max = None
                if self._sigma_max is not None:
                    ast(f'fwhm = {self._fwhm_max}')
                    sig_max = ast(fwhm_factor[peak_models[i]])
                    self._sigma_max[i] = sig_max
                if fit_type == 'uniform':
                    self.add_model(peak_models[i], prefix=f'peak{i+1}_', parameters=(
                            {'name' : 'amplitude', 'value' : amp_init, 'min' : min_value},
                            {'name' : 'center', 'expr' : center_exprs[i], 'min' : min_value},
                            {'name' : 'sigma', 'value' : sig_init, 'min' : min_value,
                            'max' : sig_max}))
                else:
                    self.add_model('gaussian', prefix=f'peak{i+1}_', parameters=(
                            {'name' : 'amplitude', 'value' : amp_init, 'min' : min_value},
                            {'name' : 'center', 'value' : cen_init, 'min' : min_value},
                            {'name' : 'sigma', 'value' : sig_init, 'min' : min_value,
                            'max' : sig_max}))

    def _check_validity(self):
        """Check for valid fit parameter results
        """
        fit_failure = False
        index = re.compile(r'\d+')
        for parameter in self.best_parameters:
            name = parameter['name']
            if ((('amplitude' in name or 'height' in name) and parameter['value'] <= 0.0) or
                    (('sigma' in name or 'fwhm' in name) and parameter['value'] <= 0.0) or
                    ('center' in name and parameter['value'] <= 0.0) or
                    (name == 'scale_factor' and parameter['value'] <= 0.0)):
                logging.info(f'Invalid fit result for {name} ({parameter["value"]})')
                fit_failure = True
            if 'sigma' in name and self._sigma_max is not None:
                if name == 'sigma':
                    sigma_max = self._sigma_max[0]
                else:
                    sigma_max = self._sigma_max[int(index.search(name).group())-1]
                i = int(index.search(name).group())-1
                if parameter['value'] > sigma_max:
                    logging.info(f'Invalid fit result for {name} ({parameter["value"]})')
                    fit_failure = True
                elif parameter['value'] == sigma_max:
                    logging.warning(f'Edge result on for {name} ({parameter["value"]})')
            if 'fwhm' in name and self._fwhm_max is not None:
                if parameter['value'] > self._fwhm_max:
                    logging.info(f'Invalid fit result for {name} ({parameter["value"]})')
                    fit_failure = True
                elif parameter['value'] == self._fwhm_max:
                    logging.warning(f'Edge result on for {name} ({parameter["value"]})')
        return fit_failure

    def _normalize(self):
        """Normalize the data
        """
        y_min = self._y.min()
        self._norm = (y_min, self._y.max()-y_min)
        if self._norm[1] == 0.0:
            self._norm = None
        else:
            self._y = (self._y-self._norm[0])/self._norm[1]

    def _renormalize(self):
        """Renormalize the data and results
        """
        if self._norm is None:
            return
        self._y = self._norm[0]+self._norm[1]*self._y
        self._result.best_fit = self._norm[0]+self._norm[1]*self._result.best_fit
        for name in self._result.params:
            par = self._result.params[name]
            if 'amplitude' in name or 'height' in name or 'background' in name:
                par.value *= self._norm[1]
                if par.stderr is not None:
                    par.stderr *= self._norm[1]
                if par.init_value is not None:
                    par.init_value *= self._norm[1]
                if par.min is not None and not np.isinf(par.min):
                    par.min *= self._norm[1]
                if par.max is not None and not np.isinf(par.max):
                    par.max *= self._norm[1]
            if 'intercept' in name or 'backgroundc' in name:
                par.value += self._norm[0]
                if par.init_value is not None:
                    par.init_value += self._norm[0]
                if par.min is not None and not np.isinf(par.min):
                    par.min += self._norm[0]
                if par.max is not None and not np.isinf(par.max):
                    par.max += self._norm[0]
        self._result.init_fit = self._norm[0]+self._norm[1]*self._result.init_fit
        init_values = {}
        for name in self._result.init_values:
            init_values[name] = self._result.init_values[name]
            if init_values[name] is None:
                continue
            if 'amplitude' in name or 'height' in name or 'background' in name:
                init_values[name] *= self._norm[1]
            if 'intercept' in name or 'backgroundc' in name:
                init_values[name] += self._norm[0]
        self._result.init_values = init_values
        # Don't renormalized chisqr, it has no useful meaning in physical units
        #self._result.chisqr *= self._norm[1]*self._norm[1]
        if self._result.covar is not None:
            for i, name in enumerate(self._result.var_names):
                if 'amplitude' in name or 'height' in name or 'background' in name:
                    for j in range(len(self._result.var_names)):
                        if self._result.covar[i,j] is not None:
                            self._result.covar[i,j] *= self._norm[1]
                        if self._result.covar[j,i] is not None:
                            self._result.covar[j,i] *= self._norm[1]
        # Don't renormalized redchi, it has no useful meaning in physical units
        #self._result.redchi *= self._norm[1]*self._norm[1]
        self._result.residual *= self._norm[1]
