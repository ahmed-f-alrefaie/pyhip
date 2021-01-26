from pyhip.tools import context_dependent_memoize
import numpy as np


def platform_bits():
    import sys

    if sys.maxsize > 2 ** 32:
        return 64
    else:
        return 32


def has_stack():
    from pyhip.driver import Context

    return Context.get_device().compute_capability() >= (2, 0)


def has_double_support():
    from pyhip.driver import Context

    return Context.get_device().compute_capability() >= (1, 3)


@context_dependent_memoize
def sizeof(type_name, preamble=""):
    from pyhip.compiler import SourceModule

    mod = SourceModule(
        """
    %s
    extern "C"
    __global__ void write_size(size_t *output)
    {
      *output = sizeof(%s);
    }
    """
        % (preamble, type_name),
        no_extern_c=True,
    )

    import pyhip.gpuarray as gpuarray

    output = gpuarray.empty((), dtype=np.uintp)
    mod.get_function("write_size")(output, block=(1, 1, 1), grid=(1, 1))

    return int(output.get())
