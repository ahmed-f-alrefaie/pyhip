import pyhip.driver as hip
from decorator import decorator
import pyhip._driver as _drv
import numpy as np
from pyhip.compyte.dtypes import (  # noqa: F401
    register_dtype,
    get_or_register_dtype,
    _fill_dtype_registry,
    dtype_to_ctype as base_dtype_to_ctype,
)


_fill_dtype_registry(respect_windows=True)

def search_on_path(filenames):
    """Find file on system path."""
    # http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/52224

    from os.path import exists, abspath, join
    from os import pathsep, environ

    search_path = environ["PATH"]

    paths = search_path.split(pathsep)
    for path in paths:
        for filename in filenames:
            if exists(join(path, filename)):
                return abspath(join(path, filename))


def make_default_context(ctx_maker=None):
    if ctx_maker is None:

        def ctx_maker(dev):
            return dev.make_context()

    ndevices = hip.Device.count()
    if ndevices == 0:
        raise RuntimeError(
            "No CUDA enabled device found. " "Please check your installation."
        )

    # Is CUDA_DEVICE set?
    import os

    devn = os.environ.get("HIP_DEVICE")

    # Is $HOME/.cuda_device set ?
    if devn is None:
        try:
            homedir = os.environ.get("HOME")
            assert homedir is not None
            devn = open(os.path.join(homedir, ".hip_device")).read().strip()
        except Exception:
            pass

    # If either CUDA_DEVICE or $HOME/.cuda_device is set, try to use it
    if devn is not None:
        try:
            devn = int(devn)
        except TypeError:
            raise TypeError(
                "HIP device number (HIP_DEVICE or ~/.hip_device)"
                " must be an integer"
            )

        dev = hip.Device(devn)
        return ctx_maker(dev)

    # Otherwise, try to use any available device
    else:
        for devn in range(ndevices):
            dev = hip.Device(devn)
            try:
                return ctx_maker(dev)
            except hip.Error:
                pass

        raise RuntimeError(
            "make_default_context() wasn't able to create a context "
            "on any of the %d detected devices" % ndevices
        )

class Argument:
    def __init__(self, dtype, name):
        self.dtype = np.dtype(dtype)
        self.name = name

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r}, {self.dtype})"


def dtype_to_ctype(dtype, with_fp_tex_hack=False):
    if dtype is None:
        raise ValueError("dtype may not be None")

    dtype = np.dtype(dtype)


    return base_dtype_to_ctype(dtype)


class VectorArg(Argument):
    def declarator(self):
        return "{} *{}".format(dtype_to_ctype(self.dtype), self.name)

    struct_char = "P"


class ScalarArg(Argument):
    def declarator(self):
        return "{} {}".format(dtype_to_ctype(self.dtype), self.name)

    @property
    def struct_char(self):
        result = self.dtype.char
        if result == "V":
            result = "%ds" % self.dtype.itemsize

        return result


def parse_c_arg(c_arg):
    from pyhip.compyte.dtypes import parse_c_arg_backend

    return parse_c_arg_backend(c_arg, ScalarArg, VectorArg)


def get_arg_type(c_arg):
    return parse_c_arg(c_arg).struct_char


# }}}

# {{{ context-dep memoization

context_dependent_memoized_functions = []


@decorator
def context_dependent_memoize(func, *args):
    try:
        ctx_dict = func._pyhip_ctx_dep_memoize_dic
    except AttributeError:
        # FIXME: This may keep contexts alive longer than desired.
        # But I guess since the memory in them is freed, who cares.
        ctx_dict = func._pyhip_ctx_dep_memoize_dic = {}

    cur_ctx = cuda.Context.get_current()

    try:
        return ctx_dict[cur_ctx][args]
    except KeyError:
        context_dependent_memoized_functions.append(func)
        arg_dict = ctx_dict.setdefault(cur_ctx, {})
        result = func(*args)
        arg_dict[args] = result
        return result


def clear_context_caches():
    for func in context_dependent_memoized_functions:
        try:
            ctx_dict = func._pyhip_ctx_dep_memoize_dic
        except AttributeError:
            pass
        else:
            ctx_dict.clear()



def mark_hip_test(inner_f):
    def f(*args, **kwargs):
        import pyhip.driver

        # appears to be idempotent, i.e. no harm in calling it more than once
        pyhip.driver.init()

        ctx = make_default_context()
        try:
            assert isinstance(ctx.get_device().name(), str)
            assert isinstance(ctx.get_device().compute_capability(), tuple)
            assert isinstance(ctx.get_device().get_attributes(), dict)
            inner_f(*args, **kwargs)
        finally:
            ctx.pop()

            from pyhip.tools import clear_context_caches

            clear_context_caches()

            from gc import collect

            collect()

    try:
        from py.test import mark as mark_test
    except ImportError:
        return f

    return mark_test.hip(f)