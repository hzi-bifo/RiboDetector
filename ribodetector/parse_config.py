import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
# from logger import setup_logging
from ribodetector.utils import read_json, write_json


class ConfigParser:
    def __init__(self, config):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """
        # load config file and apply modification
        self.config = config

        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @classmethod
    def from_json(cls, config_json):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        # if not isinstance(args, tuple):
        #     args = args.parse_args()

        cfg_fname = Path(config_json)
        config = read_json(cfg_fname)
        # print(config)
        # print(cls(config))
        return cls(config)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]
                   ), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]
                   ), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2, logfile=None):
        handlers = [logging.StreamHandler()]
        if logfile is not None:
            handlers.append(logging.FileHandler(logfile, mode='w'))
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(
            verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logging.basicConfig(
            level=self.log_levels[verbosity],
            format='%(asctime)s : %(levelname)s  %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=handlers)
        logger = logging.getLogger(name)
        return logger

    # # setting read-only attributes
    # @property
    # def config(self):
    #     return self._config

    # @property
    # def save_dir(self):
    #     return self._save_dir

    # @property
    # def log_dir(self):
    #     return self._log_dir


# helper functions to update config dict with custom cli options
# def _update_config(config, modification):
#     if modification is None:
#         return config

#     for k, v in modification.items():
#         if v is not None:
#             _set_by_path(config, k, v)
#     return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
