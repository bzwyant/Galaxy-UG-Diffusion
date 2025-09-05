_MODELS = {}


def register_model(cls=None, *, name=None):
    """Decorator for registering model classes."""

    def _register(cls):
        if name is None:
            _name = cls.__name__
        else:
            _name = name
        if _name in _MODELS:
            raise ValueError(f"Model '{_name}' is already registered.")
        _MODELS[_name] = cls
        return cls
    
    if cls is None:
        return _register
    else:
        return _register(cls)
    

def get_model(name: str, **kwargs):
    if name not in _MODELS:
        raise NameError(f"Model '{name}' is not registered.")
    return _MODELS[name](**kwargs)


from .ddpm import DDPM
from .ncsnv2 import NCSNv2
from .ncsnpp import NCSNPP