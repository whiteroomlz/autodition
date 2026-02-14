from abc import ABCMeta
from functools import wraps

SETUP_FUNCTION_NAME = "setup"
REQUIRES_SETUP_FLAG_NAME = "_requires_setup"
SETUP_CALLED_FLAG_NAME = "_setup_called"


def requires_setup(method):
    setattr(method, REQUIRES_SETUP_FLAG_NAME, True)
    return method


class RequiresSetupMeta(type):
    def __new__(cls, name, bases, dct):
        cls_instance = super().__new__(cls, name, bases, dct)
        cls_instance._wrap_methods()
        return cls_instance

    def _wrap_methods(cls):
        setup_method = getattr(cls, SETUP_FUNCTION_NAME, None)
        if setup_method:
            setattr(cls, SETUP_FUNCTION_NAME, cls._setup_called_decorator(setup_method))

        for method_name in cls._methods_requiring_setup():
            method = getattr(cls, method_name, None)
            if method:
                setattr(cls, method_name, cls._validate_setup_decorator(method))

    @staticmethod
    def _validate_setup_decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            if getattr(self, SETUP_CALLED_FLAG_NAME, None) is None:
                self._setup_called = False

            if not self._setup_called:
                raise RuntimeError(
                    f"Setup method call required: from {self}, method {method.__name__}"
                )
            return method(self, *args, **kwargs)

        return wrapper

    @staticmethod
    def _setup_called_decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            call_result = method(self, *args, **kwargs)
            self._setup_called = True
            return call_result

        return wrapper

    def _methods_requiring_setup(cls):
        methods = set()
        for base in cls.__mro__:
            for attr_name, attr_value in base.__dict__.items():
                if callable(attr_value) and getattr(attr_value, REQUIRES_SETUP_FLAG_NAME, False):
                    methods.add(attr_name)
        return methods


class RequiresSetupABCMeta(RequiresSetupMeta, ABCMeta):
    ...
