#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pyhip_ARRAY_API

#include <vector>
#include "tools.hpp"
#include "wrap_helpers.hpp"
#include <hip.hpp>
#include <mempool.hpp>
#include <boost/python/stl_iterator.hpp>




namespace py = boost::python;




namespace
{
  class device_allocator : public pyhip::context_dependent
  {
    public:
      typedef hipDeviceptr_t pointer_type;
      typedef size_t size_type;

      bool is_deferred() const
      {
        return false;
      }

      device_allocator *copy() const
      {
        return new device_allocator(*this);
      }

      pointer_type allocate(size_type s)
      {
        pyhip::scoped_context_activation ca(get_context());
        return pyhip::mem_alloc(s);
      }

      void free(pointer_type p)
      {
        try
        {
          pyhip::scoped_context_activation ca(get_context());
          pyhip::mem_free(p);
        }
        PYHIP_CATCH_CLEANUP_ON_DEAD_CONTEXT(pooled_device_allocation);
      }

      void try_release_blocks()
      {
        pyhip::run_python_gc();
      }
  };




  class host_allocator
  {
    private:
      unsigned m_flags;

    public:
      typedef void *pointer_type;
      typedef size_t size_type;

      bool is_deferred() const
      {
        return false;
      }

      host_allocator *copy() const
      {
        return new host_allocator(*this);
      }

      host_allocator(unsigned flags=0)
        : m_flags(flags)
      { }

      pointer_type allocate(size_type s)
      {
        return pyhip::mem_host_alloc(s, m_flags);
      }

      void free(pointer_type p)
      {
        pyhip::mem_host_free(p);
      }

      void try_release_blocks()
      {
        pyhip::run_python_gc();
      }
  };




  template<class Allocator>
  class context_dependent_memory_pool : 
    public pyhip::memory_pool<Allocator>,
    public pyhip::explicit_context_dependent
  {
    protected:
      void start_holding_blocks()
      { acquire_context(); }

      void stop_holding_blocks()
      { release_context(); }
  };




  class pooled_device_allocation 
    : public pyhip::context_dependent, 
    public pyhip::pooled_allocation<context_dependent_memory_pool<device_allocator> >
  { 
    private:
      typedef 
        pyhip::pooled_allocation<context_dependent_memory_pool<device_allocator> >
        super;

    public:
      pooled_device_allocation(
          boost::shared_ptr<super::pool_type> p, super::size_type s)
        : super(p, s)
      { }

      operator hipDeviceptr_t()
      { return ptr(); }

      operator DEV_PTR()
      { return convert_hipptr(ptr()); }
  };




  pooled_device_allocation *device_pool_allocate(
      boost::shared_ptr<context_dependent_memory_pool<device_allocator> > pool,
      context_dependent_memory_pool<device_allocator>::size_type sz)
  {
    return new pooled_device_allocation(pool, sz);
  }




  PyObject *pooled_device_allocation_to_long(pooled_device_allocation const &da)
  {
#if defined(_WIN32) && defined(_WIN64)
    return PyLong_FromVoidPtr(da.ptr());
#else
    return PyLong_FromVoidPtr(da.ptr());
#endif
  }


  
  class pooled_host_allocation 
    : public pyhip::pooled_allocation<pyhip::memory_pool<host_allocator> >
  {
    private:
      typedef 
        pyhip::pooled_allocation<pyhip::memory_pool<host_allocator> >
        super;

    public:
      pooled_host_allocation(
          boost::shared_ptr<super::pool_type> p, super::size_type s)
        : super(p, s)
      { }
  };




  py::handle<> host_pool_allocate(
      boost::shared_ptr<pyhip::memory_pool<host_allocator> > pool,
      py::object shape, py::object dtype, py::object order_py)
  {
    PyArray_Descr *tp_descr;
    if (PyArray_DescrConverter(dtype.ptr(), &tp_descr) != NPY_SUCCEED)
      throw py::error_already_set();

    std::vector<npy_intp> dims;
    std::copy(
        py::stl_input_iterator<npy_intp>(shape),
        py::stl_input_iterator<npy_intp>(),
        back_inserter(dims));

    std::auto_ptr<pooled_host_allocation> alloc(
        new pooled_host_allocation( 
          pool, tp_descr->elsize*pyhip::size_from_dims(dims.size(), &dims.front())));

    NPY_ORDER order = PyArray_CORDER;
    PyArray_OrderConverter(order_py.ptr(), &order);

    int flags = 0;
    if (order == PyArray_FORTRANORDER)
      flags |= NPY_FARRAY;
    else if (order == PyArray_CORDER)
      flags |= NPY_CARRAY;
    else
      throw std::runtime_error("unrecognized order specifier");

    py::handle<> result = py::handle<>(PyArray_NewFromDescr(
        &PyArray_Type, tp_descr,
        int(dims.size()), &dims.front(), /*strides*/ NULL,
        alloc->ptr(), flags, /*obj*/NULL));

    py::handle<> alloc_py(handle_from_new_ptr(alloc.release()));
    PyArray_BASE(result.get()) = alloc_py.get();
    Py_INCREF(alloc_py.get());

    return result;
  }



  template<class Wrapper>
  void expose_memory_pool(Wrapper &wrapper)
  {
    typedef typename Wrapper::wrapped_type cl;
    wrapper
      .add_property("held_blocks", &cl::held_blocks)
      .add_property("active_blocks", &cl::active_blocks)
      .DEF_SIMPLE_METHOD(bin_number)
      .DEF_SIMPLE_METHOD(alloc_size)
      .DEF_SIMPLE_METHOD(free_held)
      .DEF_SIMPLE_METHOD(stop_holding)
      .staticmethod("bin_number")
      .staticmethod("alloc_size")
      ;
  }
}




void pyhip_expose_tools()
{

  py::def("bitlog2", pyhip::bitlog2);

  {
    typedef context_dependent_memory_pool<device_allocator> cl;

    py::class_<
      cl, boost::noncopyable, 
      boost::shared_ptr<cl> > wrapper("DeviceMemoryPool");
    wrapper
      .def("allocate", device_pool_allocate,
          py::return_value_policy<py::manage_new_object>())
      ;

    expose_memory_pool(wrapper);
  }

  {
    typedef host_allocator cl;
    py::class_<cl> wrapper("PageLockedAllocator",
        py::init<py::optional<unsigned> >());
  }

  {
    typedef pyhip::memory_pool<host_allocator> cl;

    py::class_<
      cl, boost::noncopyable, 
      boost::shared_ptr<cl> > wrapper(
          "PageLockedMemoryPool",
          py::init<py::optional<host_allocator const &> >()
          );
    wrapper
      .def("allocate", host_pool_allocate,
          (py::arg("shape"), py::arg("dtype"), py::arg("order")="C"));
      ;

    expose_memory_pool(wrapper);
  }

  {
    typedef pooled_device_allocation cl;
    py::class_<cl, boost::noncopyable>(
        "PooledDeviceAllocation", py::no_init)
      .DEF_SIMPLE_METHOD(free)
      .def("__int__", pooled_device_allocation_to_long)
      .def("__long__", pooled_device_allocation_to_long)
      .def("__index__", pooled_device_allocation_to_long)
      .def("__len__", &cl::size)
      ;

    py::implicitly_convertible<pooled_device_allocation, DEV_PTR>();
  }

  {
    typedef pooled_host_allocation cl;
    py::class_<cl, boost::noncopyable>(
        "PooledHostAllocation", py::no_init)
      .DEF_SIMPLE_METHOD(free)
      .def("__len__", &cl::size)
      ;
  }
  
}
