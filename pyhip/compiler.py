
from functools import lru_cache



@lru_cache(maxsize=10)
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



@lru_cache(maxsize=1)
def _find_hipcc():
    from .tools import search_on_path
    return search_on_path(['hipcc','hipcc.exe'])


def _new_md5():
    """
    Taken from inducer/pycuda/compiler.py
    """

    import hashlib

    return hashlib.md5()


