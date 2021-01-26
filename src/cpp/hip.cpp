#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pyhip_ARRAY_API

#include "hip.hpp"

boost::thread_specific_ptr<pyhip::context_stack> pyhip::context_stack_ptr;