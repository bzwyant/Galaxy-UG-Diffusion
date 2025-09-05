_INVERSE_SOLVERS = {}


def register_solver(cls=None, *, name=None):
    """Decorator for registering inverse solver classes."""

    def _register(cls):
        if name is None:
            _name = cls.__name__
        else:
            _name = name
        if _name in _INVERSE_SOLVERS:
            raise ValueError(f"Solver '{_name}' is already registered.")
        _INVERSE_SOLVERS[_name] = cls
        return cls
    
    if cls is None:
        return _register
    else:
        return _register(cls)
    
    
def get_solver(name: str, **kwargs):
    if name not in _INVERSE_SOLVERS:
        raise NameError(f"Solver '{name}' is not registered.")
    return _INVERSE_SOLVERS[name](**kwargs)


from .unw import LaplacianPyramid
