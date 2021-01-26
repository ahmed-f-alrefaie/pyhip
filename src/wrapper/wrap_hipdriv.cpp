#define PY_ARRAY_UNIQUE_SYMBOL pyhip_ARRAY_API

#include <hip.hpp>

#include <utility>
#include <numeric>
#include <algorithm>

#include "tools.hpp"
#include "wrap_helpers.hpp"
#include <boost/python/stl_iterator.hpp>

using namespace pyhip;
using boost::shared_ptr;



namespace
{
        py::handle<>
        HipError,
        HipMemoryError,
        HipLogicError,
        HipRuntimeError,
        HipLaunchError;




        void translate_hip_error(const pyhip::error &err)
        {
        if (err.code() == hipErrorLaunchFailure
                || err.code() == hipErrorLaunchOutOfResources
                || err.code() == hipErrorLaunchTimeOut
        )
        PyErr_SetString(HipLaunchError.get(), err.what());
        else if (err.code() == hipErrorOutOfMemory)
        PyErr_SetString(HipMemoryError.get(), err.what());
        else if (err.code() == hipErrorNoDevice
                || err.code() == hipErrorNoBinaryForGpu
                || err.code() == hipErrorNoBinaryForGpu
                || err.code() == hipErrorFileNotFound
                || err.code() == hipErrorNotReady

                || err.code() == hipErrorECCNotCorrectable

                )
        PyErr_SetString(HipRuntimeError.get(), err.what());
        else if (err.code() == hipErrorUnknown)
        PyErr_SetString(HipError.get(), err.what());
        else
        PyErr_SetString(HipLogicError.get(), err.what());
        }



        py::tuple hip_version()
        {
        return py::make_tuple(
                HIP_VERSION_MAJOR,
                HIP_VERSION_MINOR,
                HIP_VERSION_PATCH);
        };




        class host_alloc_flags { };
        class mem_host_register_flags { };


        // {{{ "python-aware" wrappers

        py::object device_get_attribute(device const &dev, hipDeviceAttribute_t attr)
        {

        if (attr == hipDeviceAttributeComputeMode)
                return py::object(hipComputeMode(dev.get_attribute(attr)));
        return py::object(dev.get_attribute(attr));
        }



        device_allocation *mem_alloc_wrap(size_t bytes)
        {
                return new device_allocation(pyhip::mem_alloc_gc(bytes));
        }

        class pointer_holder_base_wrap
           : public pointer_holder_base,
          public py::wrapper<pointer_holder_base>
        {
           public:
              DEV_PTR get_pointer() const
             {
                return this->get_override("get_pointer")();
             }
        };

        py::tuple mem_alloc_pitch_wrap(
        size_t width, size_t height, unsigned int access_size)
        {
        std::auto_ptr<device_allocation> da;
        Py_ssize_t pitch = mem_alloc_pitch(
                da, width, height, access_size);
        return py::make_tuple(
                handle_from_new_ptr(da.release()), pitch);
        }

        // {{{ memory set

        void  py_memset_d8(hipDeviceptr_t dst, unsigned char uc, size_t n )
        { PYHIP_CALL_GUARDED_THREADED(hipMemsetD8, (dst, uc, n )); }
        void  py_memset_d16(hipDeviceptr_t dst, unsigned short us, size_t n )
        { PYHIP_CALL_GUARDED_THREADED(hipMemsetD16, (dst, us, n )); }
        void  py_memset_d32(hipDeviceptr_t dst, unsigned int ui, size_t n )
        { PYHIP_CALL_GUARDED_THREADED(hipMemsetD32, (dst, ui, n )); }

 /*       void  py_memset_d2d8(hipDeviceptr_t dst, size_t dst_pitch,
        unsigned char uc, size_t width, size_t height )
        { PYHIP_CALL_GUARDED_THREADED(hipMemsetD2D8, (dst, dst_pitch, uc, width, height)); }

        void  py_memset_d2d16(hipDeviceptr_t dst, size_t dst_pitch,
        unsigned short us, size_t width, size_t height )
        { PYHIP_CALL_GUARDED_THREADED(hipMemsetD2D16, (dst, dst_pitch, us, width, height)); }

        void  py_memset_d2d32(hipDeviceptr_t dst, size_t dst_pitch,
        unsigned int ui, size_t width, size_t height )
        { PYHIP_CALL_GUARDED_THREADED(hipMemsetD2D32, (dst, dst_pitch, ui, width, height)); }
*/
        // }}}

        // {{{ memory set async

        void  py_memset_d8_async(hipDeviceptr_t dst, unsigned char uc, size_t n, py::object stream_py )
        {
        PYHIP_PARSE_STREAM_PY;
        PYHIP_CALL_GUARDED_THREADED(hipMemsetD8Async, (dst, uc, n, s_handle));
        }
        void  py_memset_d16_async(hipDeviceptr_t dst, unsigned short us, size_t n, py::object stream_py )
        {
        PYHIP_PARSE_STREAM_PY;
        PYHIP_CALL_GUARDED_THREADED(hipMemsetD16Async, (dst, us, n, s_handle));
        }
        void  py_memset_d32_async(hipDeviceptr_t dst, unsigned int ui, size_t n, py::object stream_py )
        {
        PYHIP_PARSE_STREAM_PY;
        PYHIP_CALL_GUARDED_THREADED(hipMemsetD32Async, (dst, ui, n, s_handle));
        }
/*
        void  py_memset_d2d8_async(hipDeviceptr_t dst, size_t dst_pitch,
        unsigned char uc, size_t width, size_t height, py::object stream_py )
        {
        PYHIP_PARSE_STREAM_PY;
        PYHIP_CALL_GUARDED_THREADED(hipMemsetD2D8Async, (dst, dst_pitch, uc, width, height, s_handle));
        }

        void  py_memset_d2d16_async(hipDeviceptr_t dst, size_t dst_pitch,
        unsigned short us, size_t width, size_t height, py::object stream_py )
        {
        PYHIP_PARSE_STREAM_PY;
        PYHIP_CALL_GUARDED_THREADED(hipMemsetD2D16Async, (dst, dst_pitch, us, width, height, s_handle));
        }

        void  py_memset_d2d32_async(hipDeviceptr_t dst, size_t dst_pitch,
        unsigned int ui, size_t width, size_t height, py::object stream_py )
        {
        PYHIP_PARSE_STREAM_PY;
        PYHIP_CALL_GUARDED_THREADED(hipMemsetD2D32Async, (dst, dst_pitch, ui, width, height, s_handle));
        }
*/
        // }}}

        // {{{ memory copies

        void py_memcpy_htod(hipDeviceptr_t dst, py::object src)
        {
        py_buffer_wrapper buf_wrapper;
        buf_wrapper.get(src.ptr(), PyBUF_ANY_CONTIGUOUS);

        PYHIP_CALL_GUARDED_THREADED(hipMemcpyHtoD,
                (dst, buf_wrapper.m_buf.buf, buf_wrapper.m_buf.len));
        }




        void py_memcpy_htod_async(hipDeviceptr_t dst, py::object src, py::object stream_py)
        {
        py_buffer_wrapper buf_wrapper;
        buf_wrapper.get(src.ptr(), PyBUF_ANY_CONTIGUOUS);

        PYHIP_PARSE_STREAM_PY;

        PYHIP_CALL_GUARDED_THREADED(hipMemcpyHtoDAsync,
                (dst, buf_wrapper.m_buf.buf, buf_wrapper.m_buf.len, s_handle));
        }




        void py_memcpy_dtoh(py::object dest, hipDeviceptr_t src)
        {
        py_buffer_wrapper buf_wrapper;
        buf_wrapper.get(dest.ptr(), PyBUF_ANY_CONTIGUOUS | PyBUF_WRITABLE);

        PYHIP_CALL_GUARDED_THREADED(hipMemcpyDtoH,
                (buf_wrapper.m_buf.buf, src, buf_wrapper.m_buf.len));
        }




        void py_memcpy_dtoh_async(py::object dest, hipDeviceptr_t src, py::object stream_py)
        {
        py_buffer_wrapper buf_wrapper;
        buf_wrapper.get(dest.ptr(), PyBUF_ANY_CONTIGUOUS | PyBUF_WRITABLE);

        PYHIP_PARSE_STREAM_PY;

        PYHIP_CALL_GUARDED_THREADED(hipMemcpyDtoHAsync,
                (buf_wrapper.m_buf.buf, src, buf_wrapper.m_buf.len, s_handle));
        }




        // void py_memcpy_htoa(array const &ary, unsigned int index, py::object src)
        // {
        // py_buffer_wrapper buf_wrapper;
        // buf_wrapper.get(src.ptr(), PyBUF_ANY_CONTIGUOUS);

        // PYHIP_CALL_GUARDED_THREADED(hipMemcpyHtoA,
        //         (ary.handle(), index, buf_wrapper.m_buf.buf, buf_wrapper.m_buf.len));
        // }




        // void py_memcpy_atoh(py::object dest, array const &ary, unsigned int index)
        // {
        // py_buffer_wrapper buf_wrapper;
        // buf_wrapper.get(dest.ptr(), PyBUF_ANY_CONTIGUOUS | PyBUF_WRITABLE);

        // PYHIP_CALL_GUARDED_THREADED(hipMemcpyAtoH,
        //         (buf_wrapper.m_buf.buf, ary.handle(), index, buf_wrapper.m_buf.len));
        // }




        void  py_memcpy_dtod(hipDeviceptr_t dest, hipDeviceptr_t src,
        unsigned int byte_count)
        { PYHIP_CALL_GUARDED_THREADED(hipMemcpyDtoD, (dest, src, byte_count)); }


        void  py_memcpy_dtod_async(hipDeviceptr_t dest, hipDeviceptr_t src,
        unsigned int byte_count, py::object stream_py)
        {
        PYHIP_PARSE_STREAM_PY;

        PYHIP_CALL_GUARDED_THREADED(hipMemcpyDtoDAsync,
                (dest, src, byte_count, s_handle));
        }


        module *module_from_buffer(py::object buffer, py::object py_options,
        py::object message_handler)
        {
        const char *mod_buf;
        PYHIP_BUFFER_SIZE_T len;
        if (PyObject_AsCharBuffer(buffer.ptr(), &mod_buf, &len))
                throw py::error_already_set();
        hipModule_t mod;

        // #if CUDAPP_CUDA_VERSION >= 2010
        const size_t buf_size = 32768;
        char info_buf[buf_size], error_buf[buf_size];

        std::vector<hipJitOption> options;
        std::vector<void *> option_values;

        #define ADD_OPTION_PTR(KEY, PTR) \
        { \
        options.push_back(KEY); \
        option_values.push_back(PTR); \
        }

        ADD_OPTION_PTR(hipJitOptionInfoLogBuffer, info_buf);
        ADD_OPTION_PTR(hipJitOptionInfoLogBufferSizeBytes, (void *) buf_size);
        ADD_OPTION_PTR(hipJitOptionErrorLogBuffer, error_buf);
        ADD_OPTION_PTR(hipJitOptionErrorLogBufferSizeBytes, (void *) buf_size);

        PYTHON_FOREACH(key_value, py_options)
        ADD_OPTION_PTR(
                py::extract<hipJitOption>(key_value[0]),
                (void *) py::extract<intptr_t>(key_value[1])());
        #undef ADD_OPTION

        PYHIP_PRINT_CALL_TRACE("hipModuleLoadDataEx");
        hipError_t cu_status_code; \
        cu_status_code = hipModuleLoadDataEx(&mod, mod_buf, (unsigned int) options.size(),
                const_cast<hipJitOption *>(&*options.begin()),
                const_cast<void **>(&*option_values.begin()));

        size_t info_buf_size = size_t(option_values[1]);
        size_t error_buf_size = size_t(option_values[3]);

        if (message_handler != py::object())
        message_handler(cu_status_code == hipSuccess,
                std::string(info_buf, info_buf_size),
                std::string(error_buf, error_buf_size));

        if (cu_status_code != hipSuccess)
        throw pyhip::error("hipModuleLoadDataEx", cu_status_code,
                std::string(error_buf, error_buf_size).c_str());
        // #else
        // if (py::len(py_options))
        // throw pycuda::error("module_from_buffer", CUDA_ERROR_INVALID_VALUE,
        //         "non-empty options argument only supported on CUDA 2.1 and newer");

        // CUDAPP_CALL_GUARDED(cuModuleLoadData, (&mod, mod_buf));
        // #endif

                return new module(mod);
        }

        template <class T>
        PyObject *mem_obj_to_long(T const &mo)
        {
        #if defined(_WIN32) && defined(_WIN64)
                return PyLong_FromUnsignedLongLong((unsigned long long) mo);
        #else
                return PyLong_FromVoidPtr((hipDeviceptr_t) mo);
        #endif
        }

        template <class Allocation>
        py::handle<> numpy_empty(py::object shape, py::object dtype,
                py::object order_py, unsigned par1)
        {
                PyArray_Descr *tp_descr;
                if (PyArray_DescrConverter(dtype.ptr(), &tp_descr) != NPY_SUCCEED)
                        throw py::error_already_set();

                py::extract<npy_intp> shape_as_int(shape);
                std::vector<npy_intp> dims;

                if (shape_as_int.check())
                        dims.push_back(shape_as_int());
                else
                        std::copy(
                                py::stl_input_iterator<npy_intp>(shape),
                                py::stl_input_iterator<npy_intp>(),
                                back_inserter(dims));

                std::auto_ptr<Allocation> alloc(
                        new Allocation(
                        tp_descr->elsize*pyhip::size_from_dims(dims.size(), &dims.front()),
                        par1)
                        );

                NPY_ORDER order = PyArray_CORDER;
                PyArray_OrderConverter(order_py.ptr(), &order);

                int ary_flags = 0;
                if (order == PyArray_FORTRANORDER)
                        ary_flags |= NPY_FARRAY;
                else if (order == PyArray_CORDER)
                        ary_flags |= NPY_CARRAY;
                else
                        throw pyhip::error("numpy_empty", hipErrorInvalidValue,
                                "unrecognized order specifier");

                py::handle<> result = py::handle<>(PyArray_NewFromDescr(
                        &PyArray_Type, tp_descr,
                        int(dims.size()), &dims.front(), /*strides*/ NULL,
                        alloc->data(), ary_flags, /*obj*/NULL));

                py::handle<> alloc_py(handle_from_new_ptr(alloc.release()));
                PyArray_BASE(result.get()) = alloc_py.get();
                Py_INCREF(alloc_py.get());

                return result;
        }

        py::handle<> register_host_memory(py::object ary, unsigned flags)
        {
        if (!PyArray_Check(ary.ptr()))
        throw pyhip::error("register_host_memory", hipErrorInvalidValue,
                "ary argument is not a numpy array");

        if (!PyArray_ISCONTIGUOUS(ary.ptr()))
        throw pyhip::error("register_host_memory", hipErrorInvalidValue,
                "ary argument is not contiguous");

        std::auto_ptr<registered_host_memory> regmem(
                new registered_host_memory(
                PyArray_DATA(ary.ptr()), PyArray_NBYTES(ary.ptr()), flags, ary));

        PyObject *new_array_ptr = PyArray_FromInterface(ary.ptr());
        if (new_array_ptr == Py_NotImplemented)
        throw pyhip::error("register_host_memory", hipErrorInvalidValue,
                "ary argument does not expose array interface");

        py::handle<> result(new_array_ptr);

        py::handle<> regmem_py(handle_from_new_ptr(regmem.release()));
        PyArray_BASE(result.get()) = regmem_py.get();
        Py_INCREF(regmem_py.get());

        return result;
        }




        //void pyhip_expose_tools();

        static bool import_numpy_helper()
        {
        import_array1(false);
                return true;
        }

        BOOST_PYTHON_MODULE(_driver)
        {
                if (!import_numpy_helper())
                        throw py::error_already_set();

                py::def("get_version", hip_version);
                py::def("get_driver_version", pyhip::get_driver_version);


#define DECLARE_EXC(NAME, BASE) \
                        Hip##NAME = py::handle<>(PyErr_NewException("pyhip._driver." #NAME, BASE, NULL)); \
                        py::scope().attr(#NAME) = Hip##NAME;

                        {
                        DECLARE_EXC(Error, NULL);
                        DECLARE_EXC(MemoryError, HipError.get());
                        DECLARE_EXC(LogicError, HipError.get());
                        DECLARE_EXC(LaunchError, HipError.get());
                        DECLARE_EXC(RuntimeError, HipError.get());

                        py::register_exception_translator<pyhip::error>(translate_hip_error);
                        }


                py::enum_<unsigned int>("ctx_flags")
                .value("SCHED_AUTO", hipDeviceScheduleAuto)
                .value("SCHED_SPIN", hipDeviceScheduleSpin)
                .value("SCHED_YIELD", hipDeviceScheduleYield)
                .value("SCHED_MASK", hipDeviceScheduleMask)

                .value("BLOCKING_SYNC", hipDeviceScheduleBlockingSync)
                .value("SCHED_BLOCKING_SYNC", hipDeviceScheduleBlockingSync)

                .value("BLOCKING_SYNC", hipDeviceScheduleBlockingSync)
                .value("SCHED_BLOCKING_SYNC", hipDeviceScheduleBlockingSync)


                .value("MAP_HOST", hipDeviceMapHost)

                .value("LMEM_RESIZE_TO_MAX", hipDeviceLmemResizeToMax)

                // .value("FLAGS_MASK", CU_CTX_FLAGS_MASK)
                ;


                py::def("init", init,
                         py::arg("flags")=0);

                // {{{ device
                {
                typedef device cl;
                py::class_<cl>("Device", py::no_init)
                        .def("__init__", py::make_constructor(make_device))
                        //.def("__init__", py::make_constructor(make_device_from_pci_bus_id))
                        .DEF_SIMPLE_METHOD(count)
                        .staticmethod("count")
                        .DEF_SIMPLE_METHOD(name)

                        .DEF_SIMPLE_METHOD(pci_bus_id)
                        .DEF_SIMPLE_METHOD(compute_capability)
                        .DEF_SIMPLE_METHOD(total_memory)
                        .def("get_attribute", device_get_attribute)
                        .def(py::self == py::self)
                        .def(py::self != py::self)
                        .def("__hash__", &cl::hash)
                        .def("make_context", &cl::make_context,
                                (py::args("self"), py::args("flags")=0))
 
                        .def("retain_primary_context", &cl::retain_primary_context,
                                (py::args("self")))
                ;
                }
                // }}}

                // {{{ context
                {
                typedef context cl;
                py::class_<cl, shared_ptr<cl>, boost::noncopyable >("Context", py::no_init)
                .def(py::self == py::self)
                .def(py::self != py::self)
                .def("__hash__", &cl::hash)

                .def("attach", &cl::attach, (py::arg("flags")=0))
                .staticmethod("attach")
                .DEF_SIMPLE_METHOD(detach)


                .def("push", context_push)
                .DEF_SIMPLE_METHOD(pop)
                .staticmethod("pop")
                .DEF_SIMPLE_METHOD(get_device)
                .staticmethod("get_device")

                .DEF_SIMPLE_METHOD(synchronize)
                .staticmethod("synchronize")

                .def("get_current", (boost::shared_ptr<cl> (*)()) &cl::current_context)
                .staticmethod("get_current")

                // #if CUDAPP_CUDA_VERSION >= 3010
                // .DEF_SIMPLE_METHOD(set_limit)
                // .staticmethod("set_limit")
                // .DEF_SIMPLE_METHOD(get_limit)
                // .staticmethod("get_limit")
                // #endif

                .DEF_SIMPLE_METHOD(get_cache_config)
                .staticmethod("get_cache_config")
                .DEF_SIMPLE_METHOD(set_cache_config)
                .staticmethod("set_cache_config")
                .DEF_SIMPLE_METHOD(get_api_version)

                .def("enable_peer_access", &cl::enable_peer_access,
                        (py::arg("peer"), py::arg("flags")=0))
                .staticmethod("enable_peer_access")
                .DEF_SIMPLE_METHOD(disable_peer_access)
                .staticmethod("disable_peer_access")

                .DEF_SIMPLE_METHOD(get_shared_config)
                .staticmethod("get_shared_config")
                .DEF_SIMPLE_METHOD(set_shared_config)
                .staticmethod("set_shared_config")
                .add_property("handle", &cl::handle_int)
                ;
                }
                // }}}

                // {{{ stream
                {
                typedef stream cl;
                py::class_<cl, boost::noncopyable, shared_ptr<cl> >
                ("Stream", py::init<unsigned int>(py::arg("flags")=0))
                .DEF_SIMPLE_METHOD(synchronize)
                .DEF_SIMPLE_METHOD(is_done)

                .DEF_SIMPLE_METHOD(wait_for_event)
                .add_property("handle", &cl::handle_int)
                ;
                }
                // }}}

                // {{{ module
                {
                typedef module cl;
                py::class_<cl, boost::noncopyable, shared_ptr<cl> >("Module", py::no_init)
                .def("get_function", &cl::get_function, (py::args("self", "name")),
                        py::with_custodian_and_ward_postcall<0, 1>())
                .def("get_global", &cl::get_global, (py::args("self", "name")))
                // .def("get_texref", module_get_texref,
                //         (py::args("self", "name")),
                //         py::return_value_policy<py::manage_new_object>())
                // #if CUDAPP_CUDA_VERSION >= 3010
                // .def("get_surfref", module_get_surfref,
                //         (py::args("self", "name")),
                //         py::return_value_policy<py::manage_new_object>())
                // #endif
                ;
                }

                py::def("module_from_file", module_from_file, (py::arg("filename")),
                py::return_value_policy<py::manage_new_object>());
                py::def("module_from_buffer", module_from_buffer,
                (py::arg("buffer"),
                py::arg("options")=py::list(),
                py::arg("message_handler")=py::object()),
                py::return_value_policy<py::manage_new_object>());

                {
                typedef function cl;
                py::class_<cl>("Function", py::no_init)

                .DEF_SIMPLE_METHOD(get_attribute)

                .DEF_SIMPLE_METHOD(set_cache_config)

                .def("_launch_kernel", &cl::launch_kernel)
                ;
                }

        {
        typedef pointer_holder_base cl;
        py::class_<pointer_holder_base_wrap, boost::noncopyable>(
                "PointerHolderBase")
       .def("get_pointer", py::pure_virtual(&cl::get_pointer))
        .def("as_buffer", &cl::as_buffer,
                (py::arg("size"), py::arg("offset")=0))
        .def("__int__", &cl::operator DEV_PTR)
        .def("__long__", mem_obj_to_long<cl>)
        .def("__index__", mem_obj_to_long<cl>)
        ;

        py::implicitly_convertible<pointer_holder_base, hipDeviceptr_t>();
        }

        {
        typedef device_allocation cl;
        py::class_<cl, boost::noncopyable>("DeviceAllocation", py::no_init)
        .def("__int__", &cl::operator DEV_PTR)
        .def("__long__", mem_obj_to_long<cl>)
        .def("__index__", mem_obj_to_long<cl>)
        .def("as_buffer", &cl::as_buffer,
                (py::arg("size"), py::arg("offset")=0))
        .DEF_SIMPLE_METHOD(free)
        ;

        py::implicitly_convertible<device_allocation, hipDeviceptr_t>();
        }
/*

        {
        typedef host_pointer cl;
        py::class_<cl, boost::noncopyable>("HostPointer", py::no_init)
        .DEF_SIMPLE_METHOD(get_device_pointer)

        ;
        }

        {
        typedef pagelocked_host_allocation cl;
        py::class_<cl, boost::noncopyable, py::bases<host_pointer> > wrp(
                "PagelockedHostAllocation", py::no_init);

        wrp
        .DEF_SIMPLE_METHOD(free)

        .DEF_SIMPLE_METHOD(get_flags)

        ;
        py::scope().attr("HostAllocation") = wrp;
        }


        {
        typedef aligned_host_allocation cl;
        py::class_<cl, boost::noncopyable, py::bases<host_pointer> > wrp(
                "AlignedHostAllocation", py::no_init);

        wrp
        .DEF_SIMPLE_METHOD(free)
        ;
        }


        {
        typedef managed_allocation cl;
        py::class_<cl, boost::noncopyable, py::bases<device_allocation> > wrp(
                "ManagedAllocation", py::no_init);

        wrp
        .DEF_SIMPLE_METHOD(get_device_pointer)
        .def("attach", &cl::attach,
                (py::arg("mem_flags"), py::arg("stream")=py::object()))
        ;
        }


        {
        typedef registered_host_memory cl;
        py::class_<cl, boost::noncopyable, py::bases<host_pointer> >(
                "RegisteredHostMemory", py::no_init)
        .def("unregister", &cl::free)
        ;
        }
*/

        py::def("pagelocked_empty", numpy_empty<pagelocked_host_allocation>,
        (py::arg("shape"), py::arg("dtype"), py::arg("order")="C",
        py::arg("mem_flags")=0));

        py::def("aligned_empty", numpy_empty<aligned_host_allocation>,
        (py::arg("shape"), py::arg("dtype"),
        py::arg("order")="C", py::arg("alignment")=4096));


        py::def("managed_empty", numpy_empty<managed_allocation>,
        (py::arg("shape"), py::arg("dtype"), py::arg("order")="C",
        py::arg("mem_flags")=0));



        py::def("register_host_memory", register_host_memory,
        (py::arg("ary"), py::arg("flags")=0));


        // }}}

        DEF_SIMPLE_FUNCTION(mem_get_info);
        py::def("mem_alloc", mem_alloc_wrap,
        py::return_value_policy<py::manage_new_object>());
        py::def("mem_alloc_pitch", mem_alloc_pitch_wrap,
        py::args("width", "height", "access_size"));
        DEF_SIMPLE_FUNCTION(mem_get_address_range);

        // {{{ memset/memcpy
        py::def("memset_d8",  py_memset_d8, py::args("dest", "data", "size"));
        py::def("memset_d16", py_memset_d16, py::args("dest", "data", "size"));
        py::def("memset_d32", py_memset_d32, py::args("dest", "data", "size"));

        // py::def("memset_d2d8", py_memset_d2d8,
        // py::args("dest", "pitch", "data", "width", "height"));
        // py::def("memset_d2d16", py_memset_d2d16,
        // py::args("dest", "pitch", "data", "width", "height"));
        // py::def("memset_d2d32", py_memset_d2d32,
        // py::args("dest", "pitch", "data", "width", "height"));

        py::def("memset_d8_async",  py_memset_d8_async,
        (py::args("dest", "data", "size"), py::arg("stream")=py::object()));
        py::def("memset_d16_async", py_memset_d16_async,
        (py::args("dest", "data", "size"), py::arg("stream")=py::object()));
        py::def("memset_d32_async", py_memset_d32_async,
        (py::args("dest", "data", "size"), py::arg("stream")=py::object()));

        // py::def("memset_d2d8_async", py_memset_d2d8_async,
        // (py::args("dest", "pitch", "data", "width", "height"),
        // py::arg("stream")=py::object()));
        // py::def("memset_d2d16_async", py_memset_d2d16_async,
        // (py::args("dest", "pitch", "data", "width", "height"),
        // py::arg("stream")=py::object()));
        // py::def("memset_d2d32_async", py_memset_d2d32_async,
        // (py::args("dest", "pitch", "data", "width", "height"),
        // py::arg("stream")=py::object()));

        py::def("memcpy_htod", py_memcpy_htod,
        (py::args("dest"), py::arg("src")));
        py::def("memcpy_htod_async", py_memcpy_htod_async,
        (py::args("dest"), py::arg("src"), py::arg("stream")=py::object()));
        py::def("memcpy_dtoh", py_memcpy_dtoh,
        (py::args("dest"), py::arg("src")));
        py::def("memcpy_dtoh_async", py_memcpy_dtoh_async,
        (py::args("dest"), py::arg("src"), py::arg("stream")=py::object()));

        py::def("memcpy_dtod", py_memcpy_dtod, py::args("dest", "src", "size"));

        py::def("memcpy_dtod_async", py_memcpy_dtod_async,
        (py::args("dest", "src", "size"), py::arg("stream")=py::object()));

        // #if CUDAPP_CUDA_VERSION >= 4000
        // py::def("memcpy_peer", py_memcpy_peer,
        // (py::args("dest", "src", "size"),
        // py::arg("dest_context")=py::object(),
        // py::arg("src_context")=py::object()));

        // py::def("memcpy_peer_async", py_memcpy_peer_async,
        // (py::args("dest", "src", "size"),
        // py::arg("dest_context")=py::object(),
        // py::arg("src_context")=py::object(),
        // py::arg("stream")=py::object()));
        // #endif

        // DEF_SIMPLE_FUNCTION_WITH_ARGS(memcpy_dtoa,
        // ("ary", "index", "src", "len"));
        // DEF_SIMPLE_FUNCTION_WITH_ARGS(memcpy_atod,
        // ("dest", "ary", "index", "len"));
        // DEF_SIMPLE_FUNCTION_WITH_ARGS(py_memcpy_htoa,
        // ("ary", "index", "src"));
        // DEF_SIMPLE_FUNCTION_WITH_ARGS(py_memcpy_atoh,
        // ("dest", "ary", "index"));

        // DEF_SIMPLE_FUNCTION_WITH_ARGS(memcpy_atoa,
        // ("dest", "dest_index", "src", "src_index", "len"));

        // #if CUDAPP_CUDA_VERSION >= 4000
        // #define WRAP_MEMCPY_2D_UNIFIED_SETTERS \
        // .DEF_SIMPLE_METHOD(set_src_unified) \
        // .DEF_SIMPLE_METHOD(set_dst_unified)
        // #else
        // #define WRAP_MEMCPY_2D_UNIFIED_SETTERS /* empty */
        // #endif

        // #define WRAP_MEMCPY_2D_PROPERTIES \
        // .def_readwrite("src_x_in_bytes", &cl::srcXInBytes) \
        // .def_readwrite("src_y", &cl::srcY) \
        // .def_readwrite("src_memory_type", &cl::srcMemoryType) \
        // .def_readwrite("src_device", &cl::srcDevice) \
        // .def_readwrite("src_pitch", &cl::srcPitch) \
        // \
        // .DEF_SIMPLE_METHOD(set_src_host) \
        // .DEF_SIMPLE_METHOD(set_src_array) \
        // .DEF_SIMPLE_METHOD(set_src_device) \
        // \
        // .def_readwrite("dst_x_in_bytes", &cl::dstXInBytes) \
        // .def_readwrite("dst_y", &cl::dstY) \
        // .def_readwrite("dst_memory_type", &cl::dstMemoryType) \
        // .def_readwrite("dst_device", &cl::dstDevice) \
        // .def_readwrite("dst_pitch", &cl::dstPitch) \
        // \
        // .DEF_SIMPLE_METHOD(set_dst_host) \
        // .DEF_SIMPLE_METHOD(set_dst_array) \
        // .DEF_SIMPLE_METHOD(set_dst_device) \
        // \
        // .def_readwrite("width_in_bytes", &cl::WidthInBytes) \
        // .def_readwrite("height", &cl::Height) \
        // \
        // WRAP_MEMCPY_2D_UNIFIED_SETTERS

        // {
        // typedef memcpy_2d cl;
        // py::class_<cl>("Memcpy2D")
        // WRAP_MEMCPY_2D_PROPERTIES

        // .def("__call__", &cl::execute, py::args("self", "aligned"))
        // .def("__call__", &cl::execute_async)
        // ;
        // }

        // #if CUDAPP_CUDA_VERSION >= 2000
        // #define WRAP_MEMCPY_3D_PROPERTIES \
        // WRAP_MEMCPY_2D_PROPERTIES \
        // .def_readwrite("src_z", &cl::srcZ) \
        // .def_readwrite("src_lod", &cl::srcLOD) \
        // .def_readwrite("src_height", &cl::srcHeight) \
        // \
        // .def_readwrite("dst_z", &cl::dstZ) \
        // .def_readwrite("dst_lod", &cl::dstLOD) \
        // .def_readwrite("dst_height", &cl::dstHeight) \
        // \
        // .def_readwrite("depth", &cl::Depth) \

        // {
        // typedef memcpy_3d cl;
        // py::class_<cl>("Memcpy3D")
        // WRAP_MEMCPY_3D_PROPERTIES

        // .def("__call__", &cl::execute)
        // .def("__call__", &cl::execute_async)
        // ;
        // }
        // #endif
        // #if CUDAPP_CUDA_VERSION >= 4000
        // {
        // typedef memcpy_3d_peer cl;
        // py::class_<cl>("Memcpy3DPeer")
        // WRAP_MEMCPY_3D_PROPERTIES

        // .DEF_SIMPLE_METHOD(set_src_context)
        // .DEF_SIMPLE_METHOD(set_dst_context)

        // .def("__call__", &cl::execute)
        // .def("__call__", &cl::execute_async)
        // ;
        // }
        // #endif
        // }}}

        // {{{ event
        {
        typedef event cl;
        py::class_<cl, boost::noncopyable>
        ("Event", py::init<py::optional<unsigned int> >(py::arg("flags")))
        .def("record", &cl::record,
                py::arg("stream")=py::object(), py::return_self<>())
        .def("synchronize", &cl::synchronize, py::return_self<>())
        .DEF_SIMPLE_METHOD(query)
        .DEF_SIMPLE_METHOD(time_since)
        .DEF_SIMPLE_METHOD(time_till)
        // #if CUDAPP_CUDA_VERSION >= 4010 && PY_VERSION_HEX >= 0x02060000
        // .DEF_SIMPLE_METHOD(ipc_handle)
        // .def("from_ipc_handle", event_from_ipc_handle,
        //         py::return_value_policy<py::manage_new_object>())
        // .staticmethod("from_ipc_handle")
        // #endif
        ;
        }
        // }}}

        // {{{ arrays
        // {
        // typedef CUDA_ARRAY_DESCRIPTOR cl;
        // py::class_<cl>("ArrayDescriptor")
        // .def_readwrite("width", &cl::Width)
        // .def_readwrite("height", &cl::Height)
        // .def_readwrite("format", &cl::Format)
        // .def_readwrite("num_channels", &cl::NumChannels)
        // ;
        // }

        // #if CUDAPP_CUDA_VERSION >= 2000
        // {
        // typedef CUDA_ARRAY3D_DESCRIPTOR cl;
        // py::class_<cl>("ArrayDescriptor3D")
        // .def_readwrite("width", &cl::Width)
        // .def_readwrite("height", &cl::Height)
        // .def_readwrite("depth", &cl::Depth)
        // .def_readwrite("format", &cl::Format)
        // .def_readwrite("num_channels", &cl::NumChannels)
        // .def_readwrite("flags", &cl::Flags)
        // ;
        // }
        // #endif

        // {
        // typedef array cl;
        // py::class_<cl, shared_ptr<cl>, boost::noncopyable>
        // ("Array", py::init<const CUDA_ARRAY_DESCRIPTOR &>())
        // .DEF_SIMPLE_METHOD(free)
        // .DEF_SIMPLE_METHOD(get_descriptor)
        // #if CUDAPP_CUDA_VERSION >= 2000
        // .def(py::init<const CUDA_ARRAY3D_DESCRIPTOR &>())
        // .DEF_SIMPLE_METHOD(get_descriptor_3d)
        // #endif
        // .add_property("handle", &cl::handle_int)
        // ;
        // }
        // // }}}

        // // {{{ texture reference
        // {
        // typedef texture_reference cl;
        // py::class_<cl, boost::noncopyable>("TextureReference")
        // .DEF_SIMPLE_METHOD(set_array)
        // .def("set_address", &cl::set_address,
        //         (py::arg("devptr"), py::arg("bytes"), py::arg("allow_offset")=false))
        // #if CUDAPP_CUDA_VERSION >= 2020
        // .DEF_SIMPLE_METHOD_WITH_ARGS(set_address_2d, ("devptr", "descr", "pitch"))
        // #endif
        // .DEF_SIMPLE_METHOD_WITH_ARGS(set_format, ("format", "num_components"))
        // .DEF_SIMPLE_METHOD_WITH_ARGS(set_address_mode, ("dim", "am"))
        // .DEF_SIMPLE_METHOD(set_filter_mode)
        // .DEF_SIMPLE_METHOD(set_flags)
        // .DEF_SIMPLE_METHOD(get_address)
        // .def("get_array", &cl::get_array,
        //         py::return_value_policy<py::manage_new_object>())
        // .DEF_SIMPLE_METHOD(get_address_mode)
        // .DEF_SIMPLE_METHOD(get_filter_mode)

        // #if CUDAPP_CUDA_VERSION >= 2000
        // .DEF_SIMPLE_METHOD(get_format)
        // #endif

        // .DEF_SIMPLE_METHOD(get_flags)
        // ;
        // }
        // // }}}

        // // {{{ surface reference
        // #if CUDAPP_CUDA_VERSION >= 3010
        // {
        // typedef surface_reference cl;
        // py::class_<cl, boost::noncopyable>("SurfaceReference", py::no_init)
        // .def("set_array", &cl::set_array,
        //         (py::arg("array"), py::arg("flags")=0))
        // .def("get_array", &cl::get_array,
        //         py::return_value_policy<py::manage_new_object>())
        // ;
        // }
        // #endif
        // // }}}

        // // {{{ profiler control
        // #if CUDAPP_CUDA_VERSION >= 4000
        // DEF_SIMPLE_FUNCTION(initialize_profiler);
        // DEF_SIMPLE_FUNCTION(start_profiler);
        // DEF_SIMPLE_FUNCTION(stop_profiler);
        // #endif
        // // }}}

        // py::scope().attr("TRSA_OVERRIDE_FORMAT") = CU_TRSA_OVERRIDE_FORMAT;
        // py::scope().attr("TRSF_READ_AS_INTEGER") = CU_TRSF_READ_AS_INTEGER;
        // py::scope().attr("TRSF_NORMALIZED_COORDINATES") = CU_TRSF_NORMALIZED_COORDINATES;
        // py::scope().attr("TR_DEFAULT") = CU_PARAM_TR_DEFAULT;

        // DEF_SIMPLE_FUNCTION(have_gl_ext);

        //pyhip_expose_tools();


        }
}
