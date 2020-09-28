def __init__(func):

    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        self._inside_context = False

    return wrapper


def __enter__(func):

    def wrapper(self, *args, **kwargs):
        self._inside_context = True
        return func(self, *args, **kwargs)

    return wrapper


def __exit__(func):

    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        self._inside_context = False

    return wrapper


def force_in_context(func):

    def wrapper(self, *args, **kwargs):
        if not self._inside_context:
            raise ValueError("This method must be called inside a context manager")
        return func(self, *args, **kwargs)

    return wrapper
