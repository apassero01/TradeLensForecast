import importlib

from django.apps import apps

class ModelSetsStrategy:
    required_keys = ['step_number', 'm_service', 'type']
    def __init__(self, config):
        self.config = config

        if 'step_number' not in config.keys():
            raise ValueError("Missing step_number in config")
        if 'm_service' not in config.keys():
            raise ValueError("Missing m_service in config")
        if 'type' not in config.keys():
            raise ValueError("Missing type in config")
        if 'parent_strategy' not in config.keys():
            raise ValueError("Missing parent_strategy in config")

        self.config['is_applied'] = False

    def apply(self, model_sets):
        raise NotImplementedError("Subclasses must implement this method")

    def get_config(self):
        return self.config

    def get_step_number(self):
        return self.config['step_number']

    def get_m_service(self):
        return self.config['m_service']

    def get_type(self):
        return self.config['type']

    @staticmethod
    def get_required_config():
        """
        Returns a dictionary with required configuration keys, all initialized to None.
        """
        return {key: None for key in ModelSetsStrategy.required_keys}

    @staticmethod
    def get_default_config():
        return {'name': ModelSetsStrategy.__class__.__name__}

class VizProcessingStrategy:
    def __init__(self, config):
        self.config = config

        if 'm_service' not in config.keys():
            raise ValueError("Missing m_service in config")
        if 'type' not in config.keys():
            raise ValueError("Missing type in config")
        if 'parent_strategy' not in config.keys():
            raise ValueError("Missing name in config")

    def apply(self, arr):
        raise NotImplementedError("Subclasses must implement this method")

    def get_config(self):
        return self.config

    @staticmethod
    def get_default_config():
        return {'name': VizProcessingStrategy.__class__.__name__}

