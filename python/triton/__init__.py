from .kernel import *
from .ops import *

# clean-up libtriton resources
import atexit
import triton._C.libtriton as libtriton
@atexit.register
def cleanup():
  libtriton.cleanup()