import logging
import os
import yaml
from functools import cache
from copy import deepcopy

from general import illegal_value, is_int, is_num, input_yesno

#from hexrd.instrument import HEDMInstrument, PlanarDetector

class DetectorConfig:
    def __init__(self, config_source):
        self._config_source = config_source

        if isinstance(self._config_source, ((str, bytes, os.PathLike, int))):
            self._config_file = self._config_source
            self._config = self._load_config_file()
        elif isinstance(self._config_source, dict):
            self._config_file = None
            self._config = self._config_source
        else:
            self._config_file = None
            self._config = False

        self._valid = self._validate()

        if not self.valid:
            logging.error(f'Cannot create a valid instance of {self.__class__.__name__} '+
                    f'from {self._config_source}')

    def __repr__(self):
        return(f'{self.__class__.__name__}({self._config_source.__repr__()})')
    def __str__(self):
        return(f'{self.__class__.__name__} generated from {self._config_source}')

    @property
    def config_file(self):
        return(self._config_file)

    @property
    def config(self):
        return(deepcopy(self._config))

    @property
    def valid(self):
        return(self._valid)

    def load_config_file(self):
        raise(NotImplementedError)

    def validate(self):
        raise(NotImplementedError)

    def _load_config_file(self):
        if not os.path.isfile(self.config_file):
            logging.error(f'{self.config_file} is not a file.')
            return(False)
        else:
            return(self.load_config_file())

    def _validate(self):
        if not self.config:
            logging.error('A configuration must be loaded prior to calling Detector._validate')
            return(False)
        else:
            return(self.validate())

    def _write_to_file(self, out_file):
        out_file = os.path.abspath(out_file)

        current_config_valid = self.validate()
        if not current_config_valid:
            write_invalid_config = input_yesno(s=f'This {self.__class__.__name__} is currently '+
                    f'invalid. Write the configuration to {out_file} anyways?', default='no')
            if not write_invalid_config:
                logging.info('In accordance with user input, the invalid configuration will '+
                        f'not be written to {out_file}')
                return 

        if os.access(out_file, os.W_OK):
            if os.path.exists(out_file):
                overwrite = input_yesno(s=f'{out_file} already exists. Overwrite?', default='no')
                if overwrite:
                    self.write_to_file(out_file)
                else:
                    logging.info(f'In accordance with user input, {out_file} will not be '+
                            'overwritten')
            else:
                self.write_to_file(out_file)
        else:
            logging.error(f'Insufficient permissions to write to {out_file}')

    def write_to_file(self, out_file):
        raise(NotImplementedError)

class YamlDetectorConfig(DetectorConfig):
    def __init__(self, config_source, validate_yaml_pars=[]):
        self._validate_yaml_pars = validate_yaml_pars
        super().__init__(config_source)

    def load_config_file(self):
        if not os.path.splitext(self._config_file)[1]:
            if os.path.isfile(f'{self._config_file}.yml'):
                self._config_file = f'{self._config_file}.yml'
            if os.path.isfile(f'{self._config_file}.yaml'):
                self._config_file = f'{self._config_file}.yaml'
        if not os.path.isfile(self._config_file):
            logging.error(f'Unable to load {self._config_file}')
            return(False)
        with open(self._config_file, 'r') as infile:
            config = yaml.safe_load(infile)
        if isinstance(config, dict):
            return(config)
        else:
            logging.error(f'Unable to load {self._config_file} as a dictionary')
            return(False)

    def validate(self):
        if not self._validate_yaml_pars:
            logging.warning('There are no required parameters provided for this detector '+
                    'configuration')
            return(True)

        def validate_nested_pars(config, validate_yaml_par):
            yaml_par_levels = validate_yaml_par.split(':')
            first_level_par = yaml_par_levels[0]
            try:
                first_level_par = int(first_level_par)
            except:
                pass
            try:
                next_level_config = config[first_level_par]
                if len(yaml_par_levels) > 1:
                    next_level_pars = ':'.join(yaml_par_levels[1:])
                    return(validate_nested_pars(next_level_config, next_level_pars))
                else:
                    return(True)
            except:
                return(False)

        pars_missing = [p for p in self._validate_yaml_pars 
                if not validate_nested_pars(self.config, p)]
        if len(pars_missing) > 0:
            logging.error(f'Missing item(s) in configuration: {", ".join(pars_missing)}')
            return(False)
        else:
            return(True)

    def write_to_file(self, out_file):
        with open(out_file, 'w') as outf:
            yaml.dump(self.config, outf)

            
class TomoDetectorConfig(YamlDetectorConfig):
    def __init__(self, config_source):
        validate_yaml_pars = ['detector',
                              'lens_magnification',
                              'detector:pixels:rows',
                              'detector:pixels:columns',
                              *[f'detector:pixels:size:{i}' for i in range(2)]]
        super().__init__(config_source, validate_yaml_pars=validate_yaml_pars)

    @property
    @cache
    def lens_magnification(self):
        lens_magnification = self.config.get('lens_magnification')
        if not isinstance(lens_magnification, (int, float)) or lens_magnification <= 0.:
            illegal_value(lens_magnification, 'lens_magnification', 'detector file')
            logging.warning('Using default lens_magnification value of 1.0')
            return(1.0)
        else:
            return(lens_magnification)

    @property
    @cache
    def pixel_size(self):
        pixel_size = self.config['detector'].get('pixels').get('size')
        if isinstance(pixel_size, (int, float)):
            if pixel_size <= 0.:
                illegal_value(pixel_size, 'pixel_size', 'detector file')
                return(None)
            pixel_size /= self.lens_magnification
        elif isinstance(pixel_size, list):
            if ((len(pixel_size) > 2) or
                    (len(pixel_size) == 2 and pixel_size[0] != pixel_size[1])):
                illegal_value(pixel_size, 'pixel size', 'detector file')
                return(None)
            elif not is_num(pixel_size[0], 0.):
                illegal_value(pixel_size, 'pixel size', 'detector file')
                return(None)
            else:
                pixel_size = pixel_size[0]/self.lens_magnification
        else:
            illegal_value(pixel_size, 'pixel size', 'detector file')
            return(None)

        return(pixel_size)

    @property
    @cache
    def dimensions(self):
        pixels = self.config['detector'].get('pixels')
        num_rows = pixels.get('rows')
        if not is_int(num_rows, 1):
            illegal_value(num_rows, 'rows', 'detector file')
            return(None)
        num_columns = pixels.get('columns')
        if not is_int(num_columns, 1):
            illegal_value(num_columns, 'columns', 'detector file')
            return(None)
        return(num_rows, num_columns)


class EDDDetectorConfig(YamlDetectorConfig):
    def __init__(self, config_source):
        validate_yaml_pars = ['num_bins',
                              'max_E',
                              # 'angle', # KLS leave this out for now -- I think it has to do with the relative geometry of sample, beam, and detector (not a property of the detector on its own), so may not belong here in the DetectorConfig object?
                              'tth_angle',
                              'slope',
                              'intercept']
        super().__init__(config_source, validate_yaml_pars=validate_yaml_pars)

    @property
    @cache
    def num_bins(self):
        try:
            num_bins = int(self.config['num_bins'])
            if num_bins <= 0:
                raise(ValueError)
            else:
                return(num_bins)
        except:
            illegal_value(self.config['num_bins'], 'num_bins')
    @property
    @cache
    def max_E(self):
        try:
            max_E = float(self.config['max_E'])
            if max_E <= 0:
                raise(ValueError)
            else:
                return(max_E)
        except:
            illegal_value(self.config['max_E'], 'max_E')
            return(None)
            
    @property
    def bin_energies(self):
        return(self.slope * np.linspace(0, self.max_E, self.num_bins, endpoint=False) + 
                self.intercept)

    @property
    def tth_angle(self):
        try:
            return(float(self.config['tth_angle']))
        except:
            illegal_value(tth_angle, 'tth_angle')
            return(None)
    @tth_angle.setter
    def tth_angle(self, value):
        try:
            self._config['tth_angle'] = float(value)
        except:
            illegal_value(value, 'tth_angle')

    @property
    def slope(self):
        try:
            return(float(self.config['slope']))
        except:
            illegal_value(slope, 'slope')
            return(None)
    @slope.setter
    def slope(self, value):
        try:
            self._config['slope'] = float(value)
        except:
            illegal_value(value, 'slope')

    @property
    def intercept(self):
        try:
            return(float(self.config['intercept']))
        except:
            illegal_value(intercept, 'intercept')
            return(None)
    @intercept.setter
    def intercept(self, value):
        try:
            self._config['intercept'] = float(value)
        except:
            illegal_value(value, 'intercept')


# class HexrdDetectorConfig(YamlDetectorConfig):
#     def __init__(self, config_source, detector_names=[]):
#         self.detector_names = detector_names
#         validate_yaml_pars_each_detector = [*[f'buffer:{i}' for i in range(2)],
#                                             'distortion:function_name',
#                                             *[f'distortion:parameters:{i}' for i in range(6)],
#                                             'pixels:columns',
#                                             'pixels:rows',
#                                             *['pixels:size:%i' % i for i in range(2)],
#                                             'saturation_level',
#                                             *[f'transform:tilt:{i}' for i in range(3)],
#                                             *[f'transform:translation:{i}' for i in range(3)]]
#         validate_yaml_pars = []
#         for detector_name in self.detector_names:
#             validate_yaml_pars += [f'detectors:{detector_name}:{par}' for par in validate_yaml_pars_each_detector]

#         super().__init__(config_source, validate_yaml_pars=validate_yaml_pars)

#     def validate(self):
#         yaml_valid = YamlDetectorConfig.validate(self)
#         if not yaml_valid:
#             return(False)
#         else:
#             hedm_instrument = HEDMInstrument(instrument_config=self.config)
#             for detector_name in self.detector_names:
#                 if detector_name in hedm_instrument.detectors:
#                     if isinstance(hedm_instrument.detectors[detector_name], PlanarDetector):
#                         continue
#                     else:
#                         return(False)
#                 else:
#                     return(False)
#             return(True)
        
# class SAXSWAXSDetectorConfig(DetectorConfig):
#     def __init__(self, config_source):
#         super().__init__(config_source)

#     @property
#     def ai(self):
#         return(self.config)
#     @ai.setter
#     def ai(self, value):
#         if isinstance(value, pyFAI.azimuthalIntegrator.AzimuthalIntegrator):
#             self.config = ai
#         else:
#             illegal_value(value, 'azimuthal integrator')

#     # pyFAI will perform its own error-checking for the mask attribute.
#     mask = property(self.ai.get_mask, self.ai,set_mask)



