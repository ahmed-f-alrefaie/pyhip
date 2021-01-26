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
