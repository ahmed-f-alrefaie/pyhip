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

#define PYHIP_PARSE_STREAM_PY \
    hipStream_t s_handle; \
    if (stream_py.ptr() != Py_None) \
    { \
      const stream &s = py::extract<const stream &>(stream_py); \
      s_handle = s.handle(); \
    } \
    else \
      s_handle = 0;

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

#define PYHIP_CATCH_CLEANUP_ON_DEAD_CONTEXT(TYPE) \
  catch (pyhip::cannot_activate_out_of_thread_context) \
  { } \
  catch (pyhip::cannot_activate_dead_context) \
  { \
    /* PyErr_Warn( \
        PyExc_UserWarning, #TYPE " in dead context was implicitly cleaned up");*/ \
  }
namespace pyhip
{

    namespace py = boost::python;

    typedef
    #if defined(_WIN32) && defined(_WIN64)
        long long
    #else
        long
    #endif
        hash_type;
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


    struct cannot_activate_out_of_thread_context : public std::logic_error
    {
        cannot_activate_out_of_thread_context(std::string const &w)
        : std::logic_error(w)
        { }
    };

    struct cannot_activate_dead_context : public std::logic_error
    {
        cannot_activate_dead_context(std::string const &w)
        : std::logic_error(w)
        { }
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

    class context;
    class primary_context;

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

            std::string pci_bus_id()
            {
                char buffer[1024];
                PYHIP_CALL_GUARDED(hipDeviceGetPCIBusId, (buffer, sizeof(buffer), m_device));
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


            hash_type hash() const
            {
                return m_device;
            }

        boost::shared_ptr<context> make_context(unsigned int flags);
        boost::shared_ptr<context> retain_primary_context();


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
                
    class context_stack;
    extern boost::thread_specific_ptr<context_stack> context_stack_ptr;

    class context_stack
    {
        /* This wrapper is necessary because we need to pop the contents
        * off the stack before we destroy each of the contexts. This, in turn,
        * is because the contexts need to be able to access the stack in order
        * to be destroyed.
        */
        private:
        typedef std::stack<boost::shared_ptr<context> > stack_t;
        typedef stack_t::value_type value_type;;
        stack_t m_stack;

        public:
        ~context_stack();

        bool empty() const
        { return m_stack.empty(); }

        value_type &top()
        { return m_stack.top(); }

        void pop()
        {
            if (m_stack.empty())
            {
            throw error("m_stack::pop", hipErrorInvalidContext,
                "cannot pop context from empty stack");
            }
            m_stack.pop();
        }

        void push(value_type v)
        { m_stack.push(v); }

        static context_stack &get()
        {
            if (context_stack_ptr.get() == 0)
            context_stack_ptr.reset(new context_stack);

            return *context_stack_ptr;
        }
    };


  class context : boost::noncopyable
  {
    protected:
      hipCtx_t m_context;
      bool m_valid;
      unsigned m_use_count;
      boost::thread::id m_thread;

    public:
      context(hipCtx_t ctx)
        : m_context(ctx), m_valid(true), m_use_count(1),
        m_thread(boost::this_thread::get_id())
      { }

      virtual ~context()
      {
        if (m_valid)
        {
          /* It's possible that we get here with a non-zero m_use_count. Since the context
           * stack holds shared_ptrs, this must mean that the context stack itself is getting
           * destroyed, which means it's ok for this context to sign off, too.
           */
          detach();
        }
      }

      hipCtx_t handle() const
      { return m_context; }

      intptr_t handle_int() const
      { return (intptr_t) m_context; }

      bool operator==(const context &other) const
      {
        return m_context == other.m_context;
      }

      bool operator!=(const context &other) const
      {
        return m_context != other.m_context;
      }

      hash_type hash() const
      {
        return hash_type(m_context) ^ hash_type(this);
      }

      boost::thread::id thread_id() const
      { return m_thread; }

      bool is_valid() const
      {
        return m_valid;
      }

      static boost::shared_ptr<context> attach(unsigned int flags)
      {
        hipCtx_t current;

        // YHIP_CALL_GUARDED(hipCtxCreate, (&current, flags));
        // boost::shared_ptr<context> result(new context(current));
        // context_stack::get().push(result);
        // return result;
      }

    protected:
      virtual void detach_internal()
      {
        // Pop it since it will be the current

        PYHIP_CALL_GUARDED_CLEANUP(hipCtxPopCurrent, (&m_context))
        PYHIP_CALL_GUARDED_CLEANUP(hipCtxDestroy, (m_context));
      }

    public:
      virtual void detach()
      {
        if (m_valid)
        {
          bool active_before_destruction = current_context().get() == this;
          if (active_before_destruction)
          {
            detach_internal();
          }
          else
          {
            if (m_thread == boost::this_thread::get_id())
            {
              PYHIP_CALL_GUARDED_CLEANUP(hipCtxPushCurrent, (m_context));
              detach_internal();
              /* pop is implicit in detach */
            }
            else
            {
              // In all likelihood, this context's managing thread has exited, and
              // therefore this context has already been deleted. No need to harp
              // on the fact that we still thought there was cleanup to do.

              // std::cerr << "pyhip WARNING: leaked out-of-thread context " << std::endl;
            }
          }

          m_valid = false;

          if (active_before_destruction)
          {
            boost::shared_ptr<context> new_active = current_context(this);
            if (new_active.get())
            {
              PYHIP_CALL_GUARDED(hipCtxPushCurrent, (new_active->m_context));
            }
          }
        }
        else
          throw error("context::detach", hipErrorInvalidContext,
              "cannot detach from invalid context");
      }

      static device get_device()
      {
        hipDevice_t dev;
        PYHIP_CALL_GUARDED(hipCtxGetDevice, (&dev));
        return device(dev);
      }



      static void prepare_context_switch()
      {
        if (!context_stack::get().empty())
        {
          hipCtx_t popped;
          PYHIP_CALL_GUARDED(hipCtxPopCurrent, (&popped));
        }
      }

      static void pop()
      {
        prepare_context_switch();
        context_stack &ctx_stack = context_stack::get();

        if (ctx_stack.empty())
        {
          throw error("context::pop", hipErrorInvalidContext,
              "cannot pop non-current context");
        }

        boost::shared_ptr<context> current = current_context();
        if (current)
          --current->m_use_count;

        ctx_stack.pop();

        current = current_context();
        if (current)
          PYHIP_CALL_GUARDED(hipCtxPushCurrent, (current_context()->m_context));
      }


      static void synchronize()
      { PYHIP_CALL_GUARDED_THREADED(hipCtxSynchronize, ()); }

      static boost::shared_ptr<context> current_context(context *except=0)
      {
        while (true)
        {
          if (context_stack::get().empty())
            return boost::shared_ptr<context>();

          boost::shared_ptr<context> result(context_stack::get().top());
          if (result.get() != except
              && result->is_valid())
          {
            // good, weak pointer didn't expire
            return result;
          }

          // context invalid, pop it and try again.
          context_stack::get().pop();
        }
      }


    //   static void set_limit(CUlimit limit, size_t value)
    //   {
    //     PYHIP_CALL_GUARDED(hipCtxSetLimit, (limit, value));
    //   }

    //   static size_t get_limit(CUlimit limit)
    //   {
    //     size_t value;
    //     PYHIP_CALL_GUARDED(hipCtxGetLimit, (&value, limit));
    //     return value;
    //   }


      static hipFuncCache_t get_cache_config()
      {
        hipFuncCache_t value;
        PYHIP_CALL_GUARDED(hipCtxGetCacheConfig, (&value));
        return value;
      }

      static void set_cache_config(hipFuncCache_t cc)
      {
        PYHIP_CALL_GUARDED(hipCtxSetCacheConfig, (cc));
      }

      unsigned int get_api_version()
      {
        int value;
        PYHIP_CALL_GUARDED(hipCtxGetApiVersion, (m_context, &value));
        return value;
      }


      static void enable_peer_access(context const &peer, unsigned int flags)
      {
        PYHIP_CALL_GUARDED(hipCtxEnablePeerAccess, (peer.handle(), flags));
      }

      static void disable_peer_access(context const &peer)
      {
        PYHIP_CALL_GUARDED(hipCtxDisablePeerAccess, (peer.handle()));
      }



      static hipSharedMemConfig get_shared_config()
      {
        hipSharedMemConfig config;
        PYHIP_CALL_GUARDED(hipCtxGetSharedMemConfig, (&config));
        return config;
      }

      static void set_shared_config(hipSharedMemConfig config)
      {
        PYHIP_CALL_GUARDED(hipCtxSetSharedMemConfig, (config));
      }


      friend class device;
      friend void context_push(boost::shared_ptr<context> ctx);
      friend class primary_context;
  };



  class primary_context : public context
  {
   protected:
      hipDevice_t m_device;

    public:
      primary_context(hipCtx_t ctx, hipDevice_t dev)
        : context (ctx), m_device(dev)
      { }

    protected:
      virtual void detach_internal()
      {
        // Primary context comes from retainPrimaryContext.
        PYHIP_CALL_GUARDED_CLEANUP(hipDevicePrimaryCtxRelease, (m_device));
      }
  };

  inline
  boost::shared_ptr<context> device::make_context(unsigned int flags)
  {
    context::prepare_context_switch();

    hipCtx_t ctx;
    PYHIP_CALL_GUARDED_THREADED(hipCtxCreate, (&ctx, flags, m_device));
    boost::shared_ptr<context> result(new context(ctx));
    context_stack::get().push(result);
    return result;
  }


  inline boost::shared_ptr<context> device::retain_primary_context()
  {
    hipCtx_t ctx;
    PYHIP_CALL_GUARDED(hipDevicePrimaryCtxRetain, (&ctx, m_device));
    boost::shared_ptr<context> result(new primary_context(ctx, m_device));
    return result;
  }

  inline
  void context_push(boost::shared_ptr<context> ctx)
  {
    context::prepare_context_switch();

    PYHIP_CALL_GUARDED(hipCtxPushCurrent, (ctx->m_context));
    context_stack::get().push(ctx);
    ++ctx->m_use_count;
  }

  inline context_stack::~context_stack()
  {
    if (!m_stack.empty())
    {
      std::cerr
        << "-------------------------------------------------------------------" << std::endl
        << "pyhip ERROR: The context stack was not empty upon module cleanup." << std::endl
        << "-------------------------------------------------------------------" << std::endl
        << "A context was still active when the context stack was being" << std::endl
        << "cleaned up. At this point in our execution, CUDA may already" << std::endl
        << "have been deinitialized, so there is no way we can finish" << std::endl
        << "cleanly. The program will be aborted now." << std::endl
        << "Use Context.pop() to avoid this problem." << std::endl
        << "-------------------------------------------------------------------" << std::endl;
      abort();
    }
  }

  class explicit_context_dependent
  {
    private:
      boost::shared_ptr<context> m_ward_context;

    public:
      void acquire_context()
      {
        m_ward_context = context::current_context();
        if (m_ward_context.get() == 0)
          throw error("explicit_context_dependent",
              hipErrorInvalidContext,
              "no currently active context?");
      }

      void release_context()
      {
        m_ward_context.reset();
      }

      boost::shared_ptr<context> get_context()
      {
        return m_ward_context;
      }
  };

  class context_dependent : public explicit_context_dependent
  {
    private:
      boost::shared_ptr<context> m_ward_context;

    public:
      context_dependent()
      { acquire_context(); }
  };


  class scoped_context_activation
  {
    private:
      boost::shared_ptr<context> m_context;
      bool m_did_switch;

    public:
      scoped_context_activation(boost::shared_ptr<context> ctx)
        : m_context(ctx)
      {
        if (!m_context->is_valid())
          throw pyhip::cannot_activate_dead_context(
              "cannot activate dead context");

        m_did_switch = context::current_context() != m_context;
        if (m_did_switch)
        {
          if (boost::this_thread::get_id() != m_context->thread_id())
            throw pyhip::cannot_activate_out_of_thread_context(
                "cannot activate out-of-thread context");

          context_push(m_context);

          throw pyhip::error("scoped_context_activation",hipErrorInvalidContext,
              "not available in CUDA < 2.0");

        }
      }

      ~scoped_context_activation()
      {

        if (m_did_switch)
          m_context->pop();

      }

  };

  class event;

  class stream : public boost::noncopyable, public context_dependent
  {
    private:
      hipStream_t m_stream;

    public:
      stream(unsigned int flags=0)
      { 
          PYHIP_CALL_GUARDED(hipStreamCreate, (&m_stream)); 
      }

      ~stream()
      {
        try
        {
          scoped_context_activation ca(get_context());
          PYHIP_CALL_GUARDED_CLEANUP(hipStreamDestroy, (m_stream));
        }
        PYHIP_CATCH_CLEANUP_ON_DEAD_CONTEXT(stream);
      }

      void synchronize()
      { PYHIP_CALL_GUARDED_THREADED(hipStreamSynchronize, (m_stream)); }

      hipStream_t handle() const
      { return m_stream; }

      intptr_t handle_int() const
      { return (intptr_t) m_stream; }

      void wait_for_event(const event &evt);

      bool is_done() const
      {
        PYHIP_PRINT_CALL_TRACE("hipStreamQuery");
        hipError_t result = hipStreamQuery(m_stream);
        switch (result)
        {
          case hipSuccess:
            return true;
          case hipErrorNotReady:
            return false;
          default:
            PYHIP_PRINT_ERROR_TRACE("hipStreamQuery", result);
            throw error("hipStreamQuery", result);
        }
      }
  };

//   class array : public boost::noncopyable, public context_dependent
//   {
//     private:
//       hipArray m_array;
//       bool m_managed;

//     public:
//       array(const HIP_ARRAY_DESCRIPTOR &descr)
//         : m_managed(true)
//       { PYHIP_CALL_GUARDED(hipArrayCreate, (&m_array, &descr)); }

//       array(const HIP_ARRAY3D_DESCRIPTOR &descr)
//         : m_managed(true)
//       { PYHIP_CALL_GUARDED(hipArray3DCreate, (&m_array, &descr)); }

//       array(hipArray ary, bool managed)
//         : m_array(ary), m_managed(managed)
//       { }

//       ~array()
//       { free(); }

//       void free()
//       {
//         if (m_managed)
//         {
//           try
//           {
//             scoped_context_activation ca(get_context());
//             PYHIP_CALL_GUARDED_CLEANUP(hipArrayDestroy, (&m_array));
//           }
//           PYHIP_CATCH_CLEANUP_ON_DEAD_CONTEXT(array);

//           m_managed = false;
//           release_context();
//         }
//       }

//       HIP_ARRAY_DESCRIPTOR get_descriptor()
//       {
//         HIP_ARRAY_DESCRIPTOR result;
//         PYHIP_CALL_GUARDED(hipArrayGetDescriptor, (&result, &m_array));
//         return result;
//       }

//       HIP_ARRAY3D_DESCRIPTOR get_descriptor_3d()
//       {
//         HIP_ARRAY3D_DESCRIPTOR result;
//         PYHIP_CALL_GUARDED(hipArray3DGetDescriptor, (&result, &m_array));
//         return result;
//       }


//       hipArray handle() const
//       { return m_array; }

//     intptr_t handle_int() const
//     { return  (intptr_t) m_array; }
//   };

  // }}}

  // {{{ texture reference
  // {{{ module
  class function;

  class module : public boost::noncopyable, public context_dependent
  {
    private:
      hipModule_t m_module;

    public:
      module(hipModule_t mod)
        : m_module(mod)
      { }

      ~module()
      {
        try
        {
          scoped_context_activation ca(get_context());
          PYHIP_CALL_GUARDED_CLEANUP(hipModuleUnload, (m_module));
        }
        PYHIP_CATCH_CLEANUP_ON_DEAD_CONTEXT(module);
      }

      hipModule_t handle() const
      { return m_module; }

      function get_function(const char *name);
      py::tuple get_global(const char *name)
      {
        hipDeviceptr_t devptr;
        size_t bytes;
        PYHIP_CALL_GUARDED(hipModuleGetGlobal, (&devptr, &bytes, m_module, name));
        return py::make_tuple(devptr, bytes);
      }
  };

  inline
  module *module_from_file(const char *filename)
  {
    hipModule_t mod;
    PYHIP_CALL_GUARDED(hipModuleLoad, (&mod, filename));
    return new module(mod);
  }


  class function
  {
    private:
      hipFunction_t m_function;
      std::string m_symbol;

    public:
      function(hipFunction_t func, std::string const &sym)
        : m_function(func), m_symbol(sym)
      { }




      int get_attribute(hipFunction_attribute attr) const
      {
        int result;
        PYHIP_CALL_GUARDED_WITH_TRACE_INFO(
            hipFuncGetAttribute, (&result, attr, m_function), m_symbol);
        return result;
      }


      void set_cache_config(hipFuncCache_t fc)
      {
        PYHIP_CALL_GUARDED_WITH_TRACE_INFO(
            hipFuncSetCacheConfig, (m_function, fc), m_symbol);
      }

      void launch_kernel(py::tuple grid_dim_py, py::tuple block_dim_py,
          py::object parameter_buffer,
          unsigned shared_mem_bytes, py::object stream_py)
      {
        const unsigned axis_count = 3;
        unsigned grid_dim[axis_count];
        unsigned block_dim[axis_count];

        for (unsigned i = 0; i < axis_count; ++i)
        {
          grid_dim[i] = 1;
          block_dim[i] = 1;
        }

        size_t gd_length = py::len(grid_dim_py);
        if (gd_length > axis_count)
          throw pyhip::error("function::launch_kernel", hipErrorInvalidHandle,
              "too many grid dimensions in kernel launch");

        for (unsigned i = 0; i < gd_length; ++i)
          grid_dim[i] = py::extract<unsigned>(grid_dim_py[i]);

        size_t bd_length = py::len(block_dim_py);
        if (bd_length > axis_count)
          throw pyhip::error("function::launch_kernel", hipErrorInvalidHandle,
              "too many block dimensions in kernel launch");

        for (unsigned i = 0; i < bd_length; ++i)
          block_dim[i] = py::extract<unsigned>(block_dim_py[i]);

        PYHIP_PARSE_STREAM_PY;

        py_buffer_wrapper par_buf_wrapper;
        par_buf_wrapper.get(parameter_buffer.ptr(), PyBUF_ANY_CONTIGUOUS);
        size_t par_len = par_buf_wrapper.m_buf.len;

        void *config[] = {
          HIP_LAUNCH_PARAM_BUFFER_POINTER, const_cast<void *>(par_buf_wrapper.m_buf.buf),
          HIP_LAUNCH_PARAM_BUFFER_SIZE, &par_len,
          HIP_LAUNCH_PARAM_END
        };

        PYHIP_CALL_GUARDED(
             hipModuleLaunchKernel, (m_function,
               grid_dim[0], grid_dim[1], grid_dim[2],
               block_dim[0], block_dim[1], block_dim[2],
               shared_mem_bytes, s_handle, 0, config
               ));
      }


// #if PYHIP_CUDA_VERSION >= 4020
//       void set_shared_config(CUsharedconfig config)
//       {
//         PYHIP_CALL_GUARDED_WITH_TRACE_INFO(
//             cuFuncSetSharedMemConfig, (m_function, config), m_symbol);
//       }
// #endif

  };

  inline
  function module::get_function(const char *name)
  {
    hipFunction_t func;
    PYHIP_CALL_GUARDED(hipModuleGetFunction, (&func, m_module, name));
    return function(func, name);
  }


    class pointer_holder_base
  {
    public:
      virtual ~pointer_holder_base() { }
      virtual hipDeviceptr_t get_pointer() const = 0;

      operator hipDeviceptr_t() const
      { return get_pointer(); }

      py::object as_buffer(size_t size, size_t offset)
      {
        return py::object(
            py::handle<>(
#if PY_VERSION_HEX >= 0x03030000
              PyMemoryView_FromMemory((char *) (get_pointer() + offset), size,
                PyBUF_WRITE)
#else /* Py2 */
              PyBuffer_FromReadWriteMemory((void *) (get_pointer() + offset), size)
#endif
              ));
      }
  };

  class device_allocation : public boost::noncopyable, public context_dependent
  {
    private:
      bool m_valid;

    protected:
      hipDeviceptr_t m_devptr;

    public:
      device_allocation(hipDeviceptr_t devptr)
        : m_valid(true), m_devptr(devptr)
      { }

      void free()
      {
        if (m_valid)
        {
          try
          {
            scoped_context_activation ca(get_context());
            mem_free(m_devptr);
          }
          PYHIP_CATCH_CLEANUP_ON_DEAD_CONTEXT(device_allocation);

          release_context();
          m_valid = false;
        }
        else
          throw pyhip::error("device_allocation::free", hipErrorInvalidHandle);
      }

      ~device_allocation()
      {
        if (m_valid)
          free();
      }

      operator hipDeviceptr_t() const
      { return m_devptr; }

      py::object as_buffer(size_t size, size_t offset)
      {
        return py::object(
            py::handle<>(
              PyMemoryView_FromMemory((char *) (m_devptr + offset), size,
                PyBUF_WRITE)

              ));
      }
  };


  class device_allocation : public boost::noncopyable, public context_dependent
  {
    private:
      bool m_valid;

    protected:
     hipDeviceptr_t m_devptr;

    public:
      device_allocation(hipDeviceptr_t devptr)
        : m_valid(true), m_devptr(devptr)
      { }

      void free()
      {
        if (m_valid)
        {
          try
          {
            scoped_context_activation ca(get_context());
            mem_free(m_devptr);
          }
          PYHIP_CATCH_CLEANUP_ON_DEAD_CONTEXT(device_allocation);

          release_context();
          m_valid = false;
        }
        else
          throw pyhip::error("device_allocation::free", hipErrorInvalidHandle);
      }

      ~device_allocation()
      {
        if (m_valid)
          free();
      }

      operator hipDeviceptr_t() const
      { return m_devptr; }

      py::object as_buffer(size_t size, size_t offset)
      {
        return py::object(
            py::handle<>(

              PyMemoryView_FromMemory((char *) (m_devptr + offset), size,
                PyBUF_WRITE)

              ));
      }
  };

  inline Py_ssize_t mem_alloc_pitch(
      std::auto_ptr<device_allocation> &da,
        unsigned int width, unsigned int height, unsigned int access_size)
  {
    hipDeviceptr_t devptr;
    size_t pitch;
    PYHIP_CALL_GUARDED(hipMemAllocPitch, (&devptr, &pitch, width, height, access_size));
    da = std::auto_ptr<device_allocation>(new device_allocation(devptr));
    return pitch;
  }

  inline
  py::tuple mem_get_address_range(hipDeviceptr_t ptr)
  {
    hipDeviceptr_t base;
    size_t size;
    PYHIP_CALL_GUARDED(hipMemGetAddressRange, (&base, &size, ptr));
    return py::make_tuple(base, size);
  }

  inline
  void memcpy_dtoa(array const &ary, unsigned int index, hipDeviceptr_t src, unsigned int len)
  { PYHIP_CALL_GUARDED_THREADED(hipMemcpyDtoA, (ary.handle(), index, src, len)); }

  inline
  void memcpy_atod(hipDeviceptr_t dst, array const &ary, unsigned int index, unsigned int len)
  { PYHIP_CALL_GUARDED_THREADED(hipMemcpyAtoD, (dst, ary.handle(), index, len)); }

  inline
  void memcpy_atoa(
      array const &dst, unsigned int dst_index,
      array const &src, unsigned int src_index,
      unsigned int len)
  { PYHIP_CALL_GUARDED_THREADED(hipMemcpyAtoA, (dst.handle(), dst_index, src.handle(), src_index, len)); }


#define MEMCPY_SETTERS \
    void set_src_host(py::object buf_py) \
    { \
      srcMemoryType = hipMemoryTypeHost; \
      py_buffer_wrapper buf_wrapper; \
      buf_wrapper.get(buf_py.ptr(), PyBUF_STRIDED_RO); \
      srcHost = buf_wrapper.m_buf.buf; \
    } \
    \
    void set_src_array(array const &ary)  \
    {  \
      srcMemoryType = hipMemoryTypeArray; \
      srcArray = ary.handle();  \
    } \
    \
    void set_src_device(hipDeviceptr_t devptr)  \
    { \
      srcMemoryType = hipMemoryTypeDevice; \
      srcDevice = devptr; \
    } \
    \
    void set_dst_host(py::object buf_py) \
    { \
      dstMemoryType = hipMemoryTypeHost; \
      py_buffer_wrapper buf_wrapper; \
      buf_wrapper.get(buf_py.ptr(), PyBUF_STRIDED); \
      dstHost = buf_wrapper.m_buf.buf; \
    } \
    \
    void set_dst_array(array const &ary) \
    { \
      dstMemoryType = hipMemoryTypeArray; \
      dstArray = ary.handle(); \
    } \
    \
    void set_dst_device(hipDeviceptr_t devptr)  \
    { \
      dstMemoryType = hipMemoryTypeDevice; \
      dstDevice = devptr; \
    }


#define MEMCPY_SETTERS_UNIFIED \
    void set_src_unified(py::object buf_py) \
    { \
      srcMemoryType = hipMemoryTypeUnified; \
      py_buffer_wrapper buf_wrapper; \
      buf_wrapper.get(buf_py.ptr(), PyBUF_ANY_CONTIGUOUS); \
      srcHost = buf_wrapper.m_buf.buf; \
    } \
    \
    void set_dst_unified(py::object buf_py) \
    { \
      dstMemoryType = hipMemoryTypeUnified; \
      py_buffer_wrapper buf_wrapper; \
      buf_wrapper.get(buf_py.ptr(), PyBUF_ANY_CONTIGUOUS | PyBUF_WRITABLE); \
      dstHost = buf_wrapper.m_buf.buf; \
    }

  struct memcpy_2d : public hip_Memcpy2D
  {
    memcpy_2d() : hip_Memcpy2D()
    {
    }

    MEMCPY_SETTERS;
    MEMCPY_SETTERS_UNIFIED;

    void execute(bool aligned=false) const
    {
      if (aligned)
      { PYHIP_CALL_GUARDED_THREADED(hipMemcpyParam2D, (this)); }
      else
      { PYHIP_CALL_GUARDED_THREADED(hipMemcpyParam2DUnaligned, (this)); }
    }

    void execute_async(const stream &s) const
    { PYHIP_CALL_GUARDED_THREADED(hipMemcpyParam2DAsync, (this, s.handle())); }
  };


  struct memcpy_3d : public HIP_MEMCPY3D
  {
    memcpy_3d() : HIP_MEMCPY3D()
    {
    }

    MEMCPY_SETTERS;
    MEMCPY_SETTERS_UNIFIED;

    void execute() const
    {
      PYHIP_CALL_GUARDED_THREADED(hipDrvMemcpy3D, (this));
    }

    void execute_async(const stream &s) const
    { PYHIP_CALL_GUARDED_THREADED(hipDrvMemcpy3DAsync, (this, s.handle())); }
  };


  // {{{ host memory
  inline void *mem_host_alloc(size_t size, unsigned flags=0)
  {
    void *m_data;

    PYHIP_CALL_GUARDED(hipHostMalloc, (&m_data, size, flags));

   /* if (flags != 0)
      throw pyhip::error("mem_host_alloc", hipErrorInvalidValue,
          "nonzero flags in mem_host_alloc not allowed in CUDA 2.1 and older");
    PYHIP_CALL_GUARDED(hipMallocHost, (&m_data, size));
*/
    return m_data;
  }

  inline void mem_host_free(void *ptr)
  {
    PYHIP_CALL_GUARDED_CLEANUP(hipFreeHost, (ptr));
  }


  inline hipDeviceptr_t mem_managed_alloc(size_t size, unsigned flags=0)
  {
    hipDeviceptr_t m_data;
    PYHIP_CALL_GUARDED(hipMallocManaged, (&m_data, size, flags));
    return m_data;
  }



  inline void *mem_host_register(void *ptr, size_t bytes, unsigned int flags=0)
  {
    PYHIP_CALL_GUARDED(hipHostRegister, (ptr, bytes, flags));
    return ptr;
  }

  inline void mem_host_unregister(void *ptr)
  {
    PYHIP_CALL_GUARDED_CLEANUP(hipHostUnregister, (ptr));
  }


  inline void *aligned_malloc(size_t size, size_t alignment, void **original_pointer)
  {
    // alignment must be a power of two.
    if ((alignment & (alignment - 1)) != 0)
      throw pyhip::error("aligned_malloc", hipErrorInvalidValue,
          "alignment must be a power of two");

    if (alignment == 0)
      throw pyhip::error("aligned_malloc", hipErrorInvalidValue,
          "alignment must non-zero");

    void *p = malloc(size + (alignment - 1));
    if (!p)
      throw pyhip::error("aligned_malloc", hipErrorOutOfMemory,
          "aligned malloc failed");

    *original_pointer = p;

    p = (void *)((((ptrdiff_t)(p)) + (alignment-1)) & -alignment);
    return p;
  }



  struct host_pointer : public boost::noncopyable, public context_dependent
  {
    protected:
      bool m_valid;
      void *m_data;

    public:
      host_pointer()
        : m_valid(false)
      { }

      host_pointer(void *ptr)
        : m_valid(true), m_data(ptr)
      { }

      virtual ~host_pointer()
      { }

      void *data()
      { return m_data; }


      hipDeviceptr_t get_device_pointer()
      {
        hipDeviceptr_t result;
        PYHIP_CALL_GUARDED(hipHostGetDevicePointer, (&result, m_data, 0));
        return result;
      }

  };

  struct pagelocked_host_allocation : public host_pointer
  {
    public:
      pagelocked_host_allocation(size_t bytesize, unsigned flags=0)
        : host_pointer(mem_host_alloc(bytesize, flags))
      { }

      /* Don't try to be clever and coalesce these in the base class.
       * Won't work: Destructors may not call virtual functions.
       */
      ~pagelocked_host_allocation()
      {
        if (m_valid)
          free();
      }

      void free()
      {
        if (m_valid)
        {
          try
          {
            scoped_context_activation ca(get_context());
            mem_host_free(m_data);
          }
          PYHIP_CATCH_CLEANUP_ON_DEAD_CONTEXT(pagelocked_host_allocation);

          release_context();
          m_valid = false;
        }
        else
          throw pyhip::error("pagelocked_host_allocation::free", hipErrorInvalidHandle);
      }

      unsigned int get_flags()
      {
        unsigned int flags;
        PYHIP_CALL_GUARDED(hipHostGetFlags, (&flags, m_data));
        return flags;
      }

  };

  struct aligned_host_allocation : public host_pointer
  {
      void *m_original_pointer;

    public:
      aligned_host_allocation(size_t size, size_t alignment)
        : host_pointer(aligned_malloc(size, alignment, &m_original_pointer))
      { }

      /* Don't try to be clever and coalesce these in the base class.
       * Won't work: Destructors may not call virtual functions.
       */
      ~aligned_host_allocation()
      {
        if (m_valid)
          free();
      }

      void free()
      {
        if (m_valid)
        {
          ::free(m_original_pointer);
          m_valid = false;
        }
        else
          throw pyhip::error("aligned_host_allocation::free", hipErrorInvalidHandle);
      }
  };


  struct managed_allocation : public device_allocation
  {
    public:
      managed_allocation(size_t bytesize, unsigned flags=0)
        : device_allocation(mem_managed_alloc(bytesize, flags))
      { }

      // The device pointer is also valid on the host
      void *data()
      { return (void *) m_devptr; }

      hipDeviceptr_t get_device_pointer()
      {
        return m_devptr;
      }

      void attach(unsigned flags, py::object stream_py)
      {
        PYHIP_PARSE_STREAM_PY;

        PYHIP_CALL_GUARDED(hipStreamAttachMemAsync, (s_handle, &m_devptr, 0, flags));
      }

  };



  struct registered_host_memory : public host_pointer
  {
    private:
      py::object m_base;

    public:
      registered_host_memory(void *p, size_t bytes, unsigned int flags=0,
          py::object base=py::object())
        : host_pointer(mem_host_register(p, bytes, flags)), m_base(base)
      {
      }

      /* Don't try to be clever and coalesce these in the base class.
       * Won't work: Destructors may not call virtual functions.
       */
      ~registered_host_memory()
      {
        if (m_valid)
          free();
      }

      void free()
      {
        if (m_valid)
        {
          try
          {
            scoped_context_activation ca(get_context());
            mem_host_unregister(m_data);
          }
          PYHIP_CATCH_CLEANUP_ON_DEAD_CONTEXT(host_allocation);

          release_context();
          m_valid = false;
        }
        else
          throw pyhip::error("registered_host_memory::free", hipErrorInvalidHandle);
      }

      py::object base() const
      {
        return m_base;
      }
  };

  // }}}

  // {{{ event
  class event : public boost::noncopyable, public context_dependent
  {
    private:
      hipEvent_t m_event;

    public:
      event(unsigned int flags=0)
      { PYHIP_CALL_GUARDED(hipEventCreate, (&m_event)); }

      event(hipEvent_t evt)
      : m_event(evt)
      { }

      ~event()
      {
        try
        {
          scoped_context_activation ca(get_context());
          PYHIP_CALL_GUARDED_CLEANUP(hipEventDestroy, (m_event));
        }
        PYHIP_CATCH_CLEANUP_ON_DEAD_CONTEXT(event);
      }

      event *record(py::object stream_py)
      {
        PYHIP_PARSE_STREAM_PY;

        PYHIP_CALL_GUARDED(hipEventRecord, (m_event, s_handle));
        return this;
      }

      hipEvent_t handle() const
      { return m_event; }

      event *synchronize()
      {
        PYHIP_CALL_GUARDED_THREADED(hipEventSynchronize, (m_event));
        return this;
      }

      bool query() const
      {
        PYHIP_PRINT_CALL_TRACE("hipEventQuery");

        hipError_t result = hipEventQuery(m_event);
        switch (result)
        {
          case hipSuccess:
            return true;
          case hipErrorNotReady:
            return false;
          default:
            PYHIP_PRINT_ERROR_TRACE("hipEventQuery", result);
            throw error("hipEventQuery", result);
        }
      }

      float time_since(event const &start)
      {
        float result;
        PYHIP_CALL_GUARDED(hipEventElapsedTime, (&result, start.m_event, m_event));
        return result;
      }

      float time_till(event const &end)
      {
        float result;
        PYHIP_CALL_GUARDED(hipEventElapsedTime, (&result, m_event, end.m_event));
        return result;
      }


  };


  inline void stream::wait_for_event(const event &evt)
  {
    PYHIP_CALL_GUARDED(hipStreamWaitEvent, (m_stream, evt.handle(), 0));
  }




}

#endif
