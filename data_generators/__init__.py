_DATA_GENERATORS = {}


def register_data_genrator(cls=None, *, name=None):
    """Decorator for registersing data generator classes."""

    def _register(cls):
        if name is None:
            _name = cls.__name__
        else:
            _name = name
        if _name in _DATA_GENERATORS:
            raise ValueError(f"Data generator '{_name}' is already registered.")
        _DATA_GENERATORS[_name] = cls
        return cls
    
    if cls is None:
        return _register
    else:
        return _register(cls)
    
    
def get_data_generator(name: str, **kwargs):
    if name not in _DATA_GENERATORS:
        raise NameError(f"Data generator '{name}' is not registered.")
    return _DATA_GENERATORS[name](**kwargs)