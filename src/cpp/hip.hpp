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

              // std::cerr << "PyCUDA WARNING: leaked out-of-thread context " << std::endl;
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
        << "PyCUDA ERROR: The context stack was not empty upon module cleanup." << std::endl
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

        // PYHIP_CALL_GUARDED(
        //     hipLaunchKernel, (m_function,
        //       grid_dim[0], grid_dim[1], grid_dim[2],
        //       block_dim[0], block_dim[1], block_dim[2],
        //       shared_mem_bytes, s_handle, 0, config
        //       ));
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


}

#endif
