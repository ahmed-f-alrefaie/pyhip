#ifndef _PYHIP_HIP_HPP
#define _PYHIP_HIP_HPP


#define PYHIP_HIP_VERSION CUDA_VERSION


#include "hip/hip_runtime.h"


#ifndef _MSC_VER
#include <stdint.h>
#endif
#include <stdexcept>
#include <boost/shared_ptr.hpp>
#include <boost/foreach.hpp>
#include <utility>
#include <stack>
#include <iostream>
#include <vector>
#include <boost/python.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/tss.hpp>
#include <boost/version.hpp>



typedef Py_ssize_t PYHIP_BUFFER_SIZE_T;



#ifdef PYHIP_TRACE_CUDA
  #define PYHIP_PRINT_CALL_TRACE(NAME) \
        std::cerr << NAME << std::endl;
  #define PYHIP_PRINT_CALL_TRACE_INFO(NAME, EXTRA_INFO) \
        std::cerr << NAME << " (" << EXTRA_INFO << ')' << std::endl;
  #define PYHIP_PRINT_ERROR_TRACE(NAME, CODE) \
        if (CODE != hipSuccess) \
      std::cerr << NAME << " failed with code " << CODE << std::endl;
#else
  #define PYHIP_PRINT_CALL_TRACE(NAME) /*nothing*/
  #define PYHIP_PRINT_CALL_TRACE_INFO(NAME, EXTRA_INFO) /*nothing*/
  #define PYHIP_PRINT_ERROR_TRACE(NAME, CODE) /*nothing*/
#endif

#define PYHIP_CALL_GUARDED_THREADED_WITH_TRACE_INFO(NAME, ARGLIST, TRACE_INFO) \
      { \
              PYHIP_PRINT_CALL_TRACE_INFO(#NAME, TRACE_INFO); \
              hipError_t cu_status_code; \
              Py_BEGIN_ALLOW_THREADS \
                cu_status_code = NAME ARGLIST; \
              Py_END_ALLOW_THREADS \
              if (cu_status_code != hipSuccess) \
                throw pyhip::error(#NAME, cu_status_code);\
            }

#define PYHIP_CALL_GUARDED_WITH_TRACE_INFO(NAME, ARGLIST, TRACE_INFO) \
      { \
              PYHIP_PRINT_CALL_TRACE_INFO(#NAME, TRACE_INFO); \
              hipError_t cu_status_code; \
              cu_status_code = NAME ARGLIST; \
              PYHIP_PRINT_ERROR_TRACE(#NAME, cu_status_code); \
              if (cu_status_code != hipSuccess) \
                throw pyhip::error(#NAME, cu_status_code);\
            }

#define PYHIP_CALL_GUARDED_THREADED(NAME, ARGLIST) \
      { \
              PYHIP_PRINT_CALL_TRACE(#NAME); \
              hipError_t cu_status_code; \
              Py_BEGIN_ALLOW_THREADS \
                cu_status_code = NAME ARGLIST; \
              Py_END_ALLOW_THREADS \
              PYHIP_PRINT_ERROR_TRACE(#NAME, cu_status_code); \
              if (cu_status_code != hipSuccess) \
                throw pyhip::error(#NAME, cu_status_code);\
            }

#define PYHIP_CALL_GUARDED(NAME, ARGLIST) \
      { \
              PYHIP_PRINT_CALL_TRACE(#NAME); \
              hipError_t cu_status_code; \
              cu_status_code = NAME ARGLIST; \
              PYHIP_PRINT_ERROR_TRACE(#NAME, cu_status_code); \
              if (cu_status_code != hipSuccess) \
                throw pyhip::error(#NAME, cu_status_code);\
            }
#define PYHIP_CALL_GUARDED_CLEANUP(NAME, ARGLIST) \
      { \
              PYHIP_PRINT_CALL_TRACE(#NAME); \
              hipError_t cu_status_code; \
              cu_status_code = NAME ARGLIST; \
              PYHIP_PRINT_ERROR_TRACE(#NAME, cu_status_code); \
              if (cu_status_code != hipSuccess) \
                std::cerr \
                  << "PyHip WARNING: a clean-up operation failed (dead context maybe?)" \
                  << std::endl \
                  << pyhip::error::make_message(#NAME, cu_status_code) \
                  << std::endl; \
            }


namespace pyhip
{

    namespace py = boost::python;


    class error : public std::runtime_error
    {
        private:
            const char *m_routine;
            hipError_t m_code;

        public:
            static std::string make_message(const char *rout, hipError_t c, const char *msg=0)
            {
                std::string result = std::string(rout);
                result += "failed: ";
                result += error_to_str(c);
                if (msg)
                {
                    result += " - ";
                    result += msg;
                }
                return result;
            }

            error(const char *rout, hipError_t c, const char *msg=0)
                : std::runtime_error(make_message(rout, c, msg)),m_routine(rout), m_code(c)
            {}

            const char* routine() const
            {
                return m_routine;
            }

            hipError_t code() const
            {
                return m_code;
            }

            bool is_out_of_memory() const
            {
                return code() == hipErrorOutOfMemory;
            }

            static const char *error_to_str(hipError_t e)
            {
                const char* errstr = hipGetErrorString(e);
                return errstr;
            
            }
    };


    class py_buffer_wrapper : public boost::noncopyable
    {
        private:
            bool m_initialized;

        public:
            Py_buffer m_buf;

            py_buffer_wrapper() 
                : m_initialized(false)
            {}

            void get(PyObject *obj, int flags)
            {
                if (PyObject_GetBuffer(obj, &m_buf, flags))
                    throw py::error_already_set();
                m_initialized = true;
            }

            virtual ~py_buffer_wrapper()
            {
                if (m_initialized)
                    PyBuffer_Release(&m_buf);
            }
    };


    inline int get_driver_version()
    {
        int result;
        PYHIP_CALL_GUARDED(hipDriverGetVersion, (&result));
        return result;
    }

    class device
    {
        private:
            hipDevice_t m_device;
        public:
            device(hipDevice_t dev) : m_device(dev)
            {}

            static int count()
            {
                int result;
                PYHIP_CALL_GUARDED(hipGetDeviceCount, (&result));
                return result;
            }

            std::string name()
            {
                char buffer[1024];
                PYHIP_CALL_GUARDED(hipDeviceGetName, (buffer, sizeof(buffer), m_device));
                return buffer;
            }

            py::tuple compute_capability()
            {
                int major, minor;
                PYHIP_CALL_GUARDED(hipDeviceComputeCapability,(&major, &minor, m_device));
                return py::make_tuple(major, minor);
            }

            size_t total_memory()
            {
                size_t bytes;
                PYHIP_CALL_GUARDED(hipDeviceTotalMem, (&bytes, m_device));
                return bytes;
            }

            int get_attribute(hipDeviceAttribute_t attr) const
            {
                int result;
                PYHIP_CALL_GUARDED(hipDeviceGetAttribute, (&result, attr, m_device));
                return result;
            }

            bool operator==(const device &other) const
            {
                return m_device == other.m_device;
            }

            bool operator!=(const device &other) const
            {
                return m_device != other.m_device;
            }

            hipDevice_t handle() const
            {
                return m_device;
            }
    };


    inline void init(unsigned int flags)
    {
        PYHIP_CALL_GUARDED_THREADED(hipInit, (flags));
    }

    inline device *make_device(int ordinal)
    {
        hipDevice_t result;
        PYHIP_CALL_GUARDED(hipDeviceGet, (&result, ordinal));
        return new device(result);
    }
                


}

#endif
