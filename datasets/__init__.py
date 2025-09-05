_DATASETS = {}

def register_dataset(cls=None, *, name=None):
    """A decorator for registering dataset classes."""
    def _register(cls):
        if name is None:
            _name = cls.__name__
        else:
            _name = name
        if _name in _DATASETS:
            raise ValueError(f"Dataset '{_name}' is already registered.")
        _DATASETS[_name] = cls
        return cls
    
    if cls is None:
        return _register
    else:
        return _register(cls)
    
            
def get_dataset(name: str, **kwargs):
    if name not in _DATASETS:
        raise NameError(f"Dataset '{name}' is not registered.")
    return _DATASETS[name](**kwargs)