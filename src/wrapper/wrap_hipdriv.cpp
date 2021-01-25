#define PY_ARRAY_UNIQUE_SYMBOL pycuda_ARRAY_API

#include <hip.hpp>

#include <utility>
#include <numeric>
#include <algorithm>

#include "tools.hpp"
#include "wrap_helpers.hpp"
#include <boost/python/stl_iterator.hpp>

using namespace pyhip;


static bool import_numpy_helper()
{
      import_array1(false);
        return true;
}

py::tuple hip_version()
{
    return py::make_tuple(
            HIP_VERSION_MAJOR,
            HIP_VERSION_MINOR,
            HIP_VERSION_PATCH);
};
namespace
{
    BOOST_PYTHON_MODULE(_driver)
    {
          if (!import_numpy_helper())
                  throw py::error_already_set();

            py::def("get_version", hip_version);
            py::def("get_driver_version", pyhip::get_driver_version);
    }
}
