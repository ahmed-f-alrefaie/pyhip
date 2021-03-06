"""
Porting inducer/pycuda/compiler.py to work on HIP
"""

from pytools import memoize

# don't import pyhip.driver here--you'll create an import loop
import os

import sys
from tempfile import mkstemp
from os import unlink


from pytools.prefork import call_capture_output


@memoize
def hipcc_version(hipcc):
    import subprocess
    cmdline = [hipcc,'--version']
    out = subprocess.run(cmdline,capture_output=True) 

    returncode = out.returncode

    if returncode != 0:
        from warnings import warn

        warn("Could not detect HIPCC version.")

        return "Unknown hipcc version"

    return out.stdout.decode('utf-8').splitlines()[0].split(':')[1].strip()



@memoize
def _find_hipcc():
    from .tools import search_on_path
    return search_on_path(['hipcc','hipcc.exe'])


def _new_md5():


    import hashlib

    return hashlib.md5()


def preprocess_source(source,options, hipcc):
    import pathlib
    handle, source_path = mkstemp(suffix=".cpp")

    outf = open(source_path, "w")
    outf.write(source)
    outf.close()
    os.close(handle)

    cmdline = [hipcc, "--preprocess"] + options + [source_path]
    # if "win32" in sys.platform:
    #     cmdline.extend(["--compiler-options", "-EP"])
    # else:
    #     cmdline.extend(["--compiler-options", "-P"])

    result, stdout, stderr = call_capture_output(cmdline, error_on_nonzero=False)

    if result != 0:
        from pyhip.driver import CompileError

        raise CompileError(
            "hipcc preprocessing of %s failed" % source_path, cmdline, stderr=stderr
        )
    
    _file_out_path = pathlib.Path(source_path).stem+'.cui'
    with open(_file_out_path, 'r') as f:
        stdout = f.read()

    # sanity check
    if len(stdout) < 0.5 * len(source):
        from pyhip.driver import CompileError

        raise CompileError(
            "hipcc preprocessing of %s failed with ridiculously "
            "small code output - likely unsupported compiler." % source_path,
            cmdline,
            stderr=stderr.decode("utf-8", "replace"),
        )

    unlink(source_path)
    unlink(_file_out_path)
    return stdout


def compile_plain(source, options, keep, hipcc, cache_dir, target="hsaco"):
    from os.path import join

    if cache_dir:
        checksum = _new_md5()

        if "#include" in source:
           checksum.update(preprocess_source(source, options, hipcc).encode("utf-8"))
        else:
            checksum.update(source.encode("utf-8"))

        for option in options:
            checksum.update(option.encode("utf-8"))
        checksum.update(hipcc_version(hipcc).encode("utf-8"))
        from pyhip.characterize import platform_bits

        checksum.update(str(platform_bits()).encode("utf-8"))

        cache_file = checksum.hexdigest()
        cache_path = join(cache_dir, cache_file + "." + target)

        try:
            cache_file = open(cache_path, "rb")
            try:
                return cache_file.read()
            finally:
                cache_file.close()

        except Exception:
            pass

    from tempfile import mkdtemp


    file_dir = mkdtemp()
    file_root = "kernel"

    cu_file_name = file_root + ".hip.cpp"
    cu_file_path = join(file_dir, cu_file_name)

    outf = open(cu_file_path, "w")
    outf.write(str(source))
    outf.close()

    if keep:
        options = options[:]
        options.append("--keep")

        print("*** compiler output in %s" % file_dir)

    cmdline = [hipcc, "--genco"] + [cu_file_name] + options + ['-o',join(file_dir, file_root + "." + target)]
    #print(cmdline)
    result, stdout, stderr = call_capture_output(
        cmdline, cwd=file_dir, error_on_nonzero=False
    )

    try:
        result_f = open(join(file_dir, file_root + "." + target), "rb")
    except OSError:
        no_output = True
    else:
        no_output = False

    if result != 0 or (no_output and (stdout or stderr)):
        if result == 0:
            from warnings import warn

            warn(
                "PyHIP: hipcc exited with status 0, but appears to have "
                "encountered an error"
            )
        from pyhip.driver import CompileError

        raise CompileError(
            "hipcc compilation of %s failed" % cu_file_path,
            cmdline,
            stdout=stdout.decode("utf-8", "replace"),
            stderr=stderr.decode("utf-8", "replace"),
        )

    if stdout or stderr:
        lcase_err_text = (stdout + stderr).decode("utf-8", "replace").lower()
        from warnings import warn

        if "demoted" in lcase_err_text or "demoting" in lcase_err_text:
            warn(
                "hipcc said it demoted types in source code it "
                "compiled--this is likely not what you want.",
                stacklevel=4,
            )
        warn(
            "The HIP compiler succeeded, but said the following:\n"
            + (stdout + stderr).decode("utf-8", "replace"),
            stacklevel=4,
        )

    result_data = result_f.read()
    result_f.close()

    if cache_dir:
        outf = open(cache_path, "wb")
        outf.write(result_data)
        outf.close()

    if not keep:
        from os import listdir, unlink, rmdir

        for name in listdir(file_dir):
            unlink(join(file_dir, name))
        rmdir(file_dir)

    return result_data

def _get_per_user_string():
    try:
        from os import getuid
    except ImportError:
        checksum = _new_md5()
        from os import environ

        checksum.update(environ["USERNAME"].encode("utf-8"))
        return checksum.hexdigest()
    else:
        return "uid%d" % getuid()

DEFAULT_HIPCC_FLAGS = [
    _flag.strip()
    for _flag in os.environ.get("PYHIP_DEFAULT_HIPCC_FLAGS", "").split()
    if _flag.strip()
]

def compile(
    source,
    hipcc="hipcc",
    options=None,
    keep=False,
    no_extern_c=False,
    arch=None,
    code=None,
    cache_dir=None,
    include_dirs=[],
    target="hsaco",
):

    # assert target in ["cubin", "ptx", "fatbin"]

    if not no_extern_c:
        source = 'extern "C" {\n%s\n}\n' % source
    source = '#include <hip/hip_runtime.h>\n\n%s' %source  
    if options is None:
        options = DEFAULT_HIPCC_FLAGS

    options = options[:]
    if arch is None:
        from pyhip.driver import Error

        try:
            from pyhip.driver import Context
            arch = 'gfx%d' % Context.get_device().properties().gcnArch  
           # arch = "sm_%d%d" % Context.get_device().compute_capability()
        except Error:
            pass

    # from pyhip.driver import CUDA_DEBUGGING

    # if CUDA_DEBUGGING:
    #     cache_dir = False
    #     keep = True
    #     options.extend(["-g", "-G"])

    if "PYHIP_CACHE_DIR" in os.environ and cache_dir is None:
        cache_dir = os.environ["PYHIP_CACHE_DIR"]

    if "PYHIP_DISABLE_CACHE" in os.environ:
        cache_dir = False

    if cache_dir is None:
        import appdirs

        cache_dir = os.path.join(
            appdirs.user_cache_dir("pyhip", "pyhip"), "compiler-cache-v1"
        )

        from os import makedirs

        try:
            makedirs(cache_dir)
        except OSError as e:
            from errno import EEXIST

            if e.errno != EEXIST:
                raise

    if arch is not None:
        options.extend(["--offload-arch=%s" % arch,])

    # if code is not None:
    #     options.extend(["-code", code])

    # if "darwin" in sys.platform and sys.maxsize == 9223372036854775807:
    #     options.append("-m64")
    # elif "win32" in sys.platform and sys.maxsize == 9223372036854775807:
    #     options.append("-m64")
    # elif "win32" in sys.platform and sys.maxsize == 2147483647:
    #     options.append("-m32")

    include_dirs = include_dirs #+ [_find_pycuda_include_path()]

    for i in include_dirs:
        options.append("-I" + i)

    return compile_plain(source, options, keep, hipcc, cache_dir, target)


class CudaModule:
    def _check_arch(self, arch):
        if arch is None:
            return
        try:
            from pyhip.driver import Context

            capability = Context.get_device().compute_capability()
            # if tuple(map(int, tuple(arch.split("_")[1]))) > capability:
            #     from warnings import warn

            #     warn(
            #         "trying to compile for a compute capability "
            #         "higher than selected GPU"
            #     )
        except Exception:
            pass

    def _bind_module(self):
        self.get_global = self.module.get_global
        # self.get_texref = self.module.get_texref
        # if hasattr(self.module, "get_surfref"):
        #     self.get_surfref = self.module.get_surfref

    def get_function(self, name):
        return self.module.get_function(name)


class SourceModule(CudaModule):
    """
    Creates a Module from a single .cu source object linked against the
    static CUDA runtime.
    """

    def __init__(
        self,
        source,
        hipcc="hipcc",
        options=None,
        keep=False,
        no_extern_c=False,
        arch=None,
        code=None,
        cache_dir=None,
        include_dirs=[],
    ):
        self._check_arch(arch)

        cubin = compile(
            source,
            hipcc,
            options,
            keep,
            no_extern_c,
            arch,
            code,
            cache_dir,
            include_dirs,
        )

        from pyhip.driver import module_from_buffer

        self.module = module_from_buffer(cubin)

        self._bind_module()
