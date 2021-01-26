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
        hipDeviceptr_t get_pointer() const
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
                return PyLong_FromUnsignedLong((unsigned long) mo);
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




        void pyhip_expose_tools();

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








        }
}
