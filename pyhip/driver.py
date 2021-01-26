import os
import numpy as np


# {{{ add cuda lib dir to Python DLL path


def _search_on_path(filenames):
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


def _add_hip_libdir_to_dll_path():
    from os.path import join, dirname

    hipcc_path = _search_on_path(["hipcc"])
    if hipcc_path is not None:
        os.add_dll_directory(dirname(hipcc_path))
        home = os.path.abspath(os.path.join(os.path.dirname(hipcc),'..'))
        lib_path = os.path.join(home,'lib')
        os.add_dll_directory(dirname(lib_path))

    #cuda_path = os.environ.get("HIP_PATH"

try:
    _add_hip_libdir_to_dll_path()
except AttributeError:
    pass


try:
    from pyhip._driver import *  # noqa
except ImportError as e:
    if "_v2" in str(e):
        from warnings import warn

        warn(
            "Failed to import the HIP driver interface, with an error "
            "message indicating that the version of your HIP header "
            "does not match the version of your HIP driver."
        )
    raise


_memoryview = memoryview
_my_bytes = bytes


class CompileError(Error):
    def __init__(self, msg, command_line, stdout=None, stderr=None):
        self.msg = msg
        self.command_line = command_line
        self.stdout = stdout
        self.stderr = stderr

    def __str__(self):
        result = self.msg
        if self.command_line:
            try:
                result += "\n[command: %s]" % (" ".join(self.command_line))
            except Exception as e:
                print(e)
        if self.stdout:
            result += "\n[stdout:\n%s]" % self.stdout
        if self.stderr:
            result += "\n[stderr:\n%s]" % self.stderr

        return result

class ArgumentHandler:
    def __init__(self, ary):
        self.array = ary
        self.dev_alloc = None

    def get_device_alloc(self):
        if self.dev_alloc is None:
            try:
                self.dev_alloc = mem_alloc_like(self.array)
            except AttributeError:
                raise TypeError(
                    "could not determine array length of '%s': unsupported array type or not an array"
                    % type(self.array)
                )
        return self.dev_alloc

    def pre_call(self, stream):
        pass


class In(ArgumentHandler):
    def pre_call(self, stream):
        if stream is not None:
            memcpy_htod(self.get_device_alloc(), self.array)
        else:
            memcpy_htod(self.get_device_alloc(), self.array)


class Out(ArgumentHandler):
    def post_call(self, stream):
        if stream is not None:
            memcpy_dtoh(self.array, self.get_device_alloc())
        else:
            memcpy_dtoh(self.array, self.get_device_alloc())


class InOut(In, Out):
    pass

def _add_functionality():
    def device_get_attributes(dev):
        result = {}

        for att_name in dir(device_attribute):
            if not att_name[0].isupper():
                continue

            att_id = getattr(device_attribute, att_name)

            try:
                att_value = dev.get_attribute(att_id)
            except LogicError as e:
                from warnings import warn

                warn(
                    "CUDA driver raised '%s' when querying '%s' on '%s'"
                    % (e, att_name, dev)
                )
            else:
                result[att_id] = att_value

        return result
    
    def device___getattr__(dev, name):
        return dev.get_attribute(getattr(device_attribute, name.upper()))

    def _build_arg_buf(args):
        handlers = []

        arg_data = []
        format = ""
        for i, arg in enumerate(args):
            if isinstance(arg, np.number):
                arg_data.append(arg)
                format += arg.dtype.char
            elif isinstance(arg, (DeviceAllocation,)): #PooledDeviceAllocation)):
                arg_data.append(int(arg))
                format += "P"
            elif isinstance(arg, ArgumentHandler):
                handlers.append(arg)
                arg_data.append(int(arg.get_device_alloc()))
                format += "P"
            # elif isinstance(arg, np.ndarray):
            #     if isinstance(arg.base, ManagedAllocationOrStub):
            #         arg_data.append(int(arg.base))
            #         format += "P"
            #     else:
            #         arg_data.append(arg)
            #         format += "%ds" % arg.nbytes
            elif isinstance(arg, np.void):
                arg_data.append(_my_bytes(_memoryview(arg)))
                format += "%ds" % arg.itemsize
            else:
                try:
                    gpudata = np.uintp(arg.gpudata)
                except AttributeError:
                    raise TypeError("invalid type on parameter #%d (0-based)" % i)
                else:
                    # for gpuarrays
                    arg_data.append(int(gpudata))
                    format += "P"

        from pycuda._pvt_struct import pack

        return handlers, pack(format, *arg_data)


    def function_call(func, *args, **kwargs):
        grid = kwargs.pop("grid", (1, 1))
        stream = kwargs.pop("stream", None)
        block = kwargs.pop("block", None)
        shared = kwargs.pop("shared", 0)
        # texrefs = kwargs.pop("texrefs", [])
        time_kernel = kwargs.pop("time_kernel", False)

        if kwargs:
            raise ValueError(
                "extra keyword arguments: %s" % (",".join(kwargs.keys()))
            )

        if block is None:
            raise ValueError("must specify block size")

        # func._set_block_shape(*block)
        handlers, arg_buf = _build_arg_buf(args)

        for handler in handlers:
            handler.pre_call(stream)

        # for texref in texrefs:
        #     func.param_set_texref(texref)

        post_handlers = [
            handler for handler in handlers if hasattr(handler, "post_call")
        ]

        if stream is None:
            if time_kernel:
                Context.synchronize()

                from time import time

                start_time = time()

            func._launch_kernel(grid, block, arg_buf, shared, None)

            if post_handlers or time_kernel:
                Context.synchronize()

                if time_kernel:
                    run_time = time() - start_time

                for handler in post_handlers:
                    handler.post_call(stream)

                if time_kernel:
                    return run_time
        else:
            assert (
                not time_kernel
            ), "Can't time the kernel on an asynchronous invocation"
            func._launch_kernel(grid, block, arg_buf, shared, stream)

            if post_handlers:
                for handler in post_handlers:
                    handler.post_call(stream)

    def function_prepare(func, arg_types, texrefs=[]):
        # func.texrefs = texrefs

        func.arg_format = ""

        for i, arg_type in enumerate(arg_types):
            if isinstance(arg_type, type) and np.number in arg_type.__mro__:
                func.arg_format += np.dtype(arg_type).char
            elif isinstance(arg_type, np.dtype):
                if arg_type.char == "V":
                    func.arg_format += "%ds" % arg_type.itemsize
                else:
                    func.arg_format += arg_type.char
            elif isinstance(arg_type, str):
                func.arg_format += arg_type
            else:
                func.arg_format += np.dtype(np.uintp).char

        return func

    def function_prepared_call(func, grid, block, *args, **kwargs):
        # if isinstance(block, tuple):
        #     func._set_block_shape(*block)
        # else:
        #     from warnings import warn

        #     warn(
        #         "Not passing the block size to prepared_call is deprecated as of "
        #         "version 2011.1.",
        #         DeprecationWarning,
        #         stacklevel=2,
        #     )
        #     args = (block,) + args

        shared_size = kwargs.pop("shared_size", 0)

        if kwargs:
            raise TypeError(
                "unknown keyword arguments: " + ", ".join(kwargs.keys())
            )

        from pycuda._pvt_struct import pack

        arg_buf = pack(func.arg_format, *args)

        func._launch_kernel(grid, block, arg_buf, shared_size, None)

    def function_prepared_timed_call(func, grid, block, *args, **kwargs):
        shared_size = kwargs.pop("shared_size", 0)
        if kwargs:
            raise TypeError(
                "unknown keyword arguments: " + ", ".join(kwargs.keys())
            )

        from pycuda._pvt_struct import pack

        arg_buf = pack(func.arg_format, *args)

        for texref in func.texrefs:
            func.param_set_texref(texref)

        start = Event()
        end = Event()

        start.record()
        func._launch_kernel(grid, block, arg_buf, shared_size, None)
        end.record()

        def get_call_time():
            end.synchronize()
            return end.time_since(start) * 1e-3

        return get_call_time

    def function_prepared_async_call(func, grid, block, stream, *args, **kwargs):
        if isinstance(block, tuple):
            func._set_block_shape(*block)
        else:
            from warnings import warn

            warn(
                "Not passing the block size to prepared_async_call is "
                "deprecated as of version 2011.1.",
                DeprecationWarning,
                stacklevel=2,
            )
            args = (stream,) + args
            stream = block

        shared_size = kwargs.pop("shared_size", 0)

        if kwargs:
            raise TypeError(
                "unknown keyword arguments: " + ", ".join(kwargs.keys())
            )

        from pycuda._pvt_struct import pack

        arg_buf = pack(func.arg_format, *args)


        func._launch_kernel(grid, block, arg_buf, shared_size, stream)

    # }}}

    def function___getattr__(self, name):

        return self.get_attribute(getattr(function_attribute, name.upper()))


    def mark_func_method_deprecated(func):
        def new_func(*args, **kwargs):
            from warnings import warn

            warn(
                "'%s' has been deprecated in version 2011.1. Please use "
                "the stateless launch interface instead." % func.__name__[1:],
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        try:
            from functools import update_wrapper
        except ImportError:
            pass
        else:
            try:
                update_wrapper(new_func, func)
            except Exception:
                # User won't see true signature. Oh well.
                pass

        return new_func

    Device.get_attributes = device_get_attributes
    Device.__getattr__ = device___getattr__

    Function.__call__ = function_call
    Function.prepare = function_prepare
    Function.prepared_call = function_prepared_call
    Function.prepared_timed_call = function_prepared_timed_call
    Function.prepared_async_call = function_prepared_async_call


_add_functionality()


# {{{ pagelocked numpy arrays


def pagelocked_zeros(shape, dtype, order="C", mem_flags=0):
    result = pagelocked_empty(shape, dtype, order, mem_flags)
    result.fill(0)
    return result


def pagelocked_empty_like(array, mem_flags=0):
    if array.flags.c_contiguous:
        order = "C"
    elif array.flags.f_contiguous:
        order = "F"
    else:
        raise ValueError("could not detect array order")

    return pagelocked_empty(array.shape, array.dtype, order, mem_flags)


def pagelocked_zeros_like(array, mem_flags=0):
    result = pagelocked_empty_like(array, mem_flags)
    result.fill(0)
    return result


# }}}


# {{{ aligned numpy arrays


def aligned_zeros(shape, dtype, order="C", alignment=4096):
    result = aligned_empty(shape, dtype, order, alignment)
    result.fill(0)
    return result


def aligned_empty_like(array, alignment=4096):
    if array.flags.c_contiguous:
        order = "C"
    elif array.flags.f_contiguous:
        order = "F"
    else:
        raise ValueError("could not detect array order")

    return aligned_empty(array.shape, array.dtype, order, alignment)


def aligned_zeros_like(array, alignment=4096):
    result = aligned_empty_like(array, alignment)
    result.fill(0)
    return result


# }}}


# {{{ managed numpy arrays (CUDA Unified Memory)


def managed_zeros(shape, dtype, order="C", mem_flags=0):
    result = managed_empty(shape, dtype, order, mem_flags)
    result.fill(0)
    return result


def managed_empty_like(array, mem_flags=0):
    if array.flags.c_contiguous:
        order = "C"
    elif array.flags.f_contiguous:
        order = "F"
    else:
        raise ValueError("could not detect array order")

    return managed_empty(array.shape, array.dtype, order, mem_flags)


def managed_zeros_like(array, mem_flags=0):
    result = managed_empty_like(array, mem_flags)
    result.fill(0)
    return result


# }}}


def mem_alloc_like(ary):
    return mem_alloc(ary.nbytes)


# {{{ array handling


def dtype_to_array_format(dtype):
    if dtype == np.uint8:
        return array_format.UNSIGNED_INT8
    elif dtype == np.uint16:
        return array_format.UNSIGNED_INT16
    elif dtype == np.uint32:
        return array_format.UNSIGNED_INT32
    elif dtype == np.int8:
        return array_format.SIGNED_INT8
    elif dtype == np.int16:
        return array_format.SIGNED_INT16
    elif dtype == np.int32:
        return array_format.SIGNED_INT32
    elif dtype == np.float32:
        return array_format.FLOAT
    else:
        raise TypeError("cannot convert dtype '%s' to array format" % dtype)




def to_device(bf_obj):
    import sys

    if sys.version_info >= (2, 7):
        bf = memoryview(bf_obj).tobytes()
    else:
        bf = buffer(bf_obj)
    result = mem_alloc(len(bf))
    memcpy_htod(result, bf)
    return result


def from_device(devptr, shape, dtype, order="C"):
    result = np.empty(shape, dtype, order)
    memcpy_dtoh(result, devptr)
    return result


def from_device_like(devptr, other_ary):
    result = np.empty_like(other_ary)
    memcpy_dtoh(result, devptr)
    return result