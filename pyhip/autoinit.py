import pyhip.driver as hip
import atexit

# Initialize HIP
hip.init()

from pyhip.tools import make_default_context  # noqa: E402

global context
context = make_default_context()
device = context.get_device()


def _finish_up():
    global context
    context.pop()
    context = None

    from pyhip.tools import clear_context_caches

    clear_context_caches()


atexit.register(_finish_up)