#!/usr/bin/env python

# -*- coding: utf-8 -*-


import os
import sys
import warnings
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
#from distutils.sysconfig import get_config_vars
#from distutils.command import build_ext as build_ext_module
#from distutils.command.build_ext import build_ext
from distutils import ccompiler
#from distutils.version import LooseVersion
import argparse
import numpy
import ctypes
from os.path import join, dirname

#from __future__ import absolute_import, print_function
from os.path import dirname, join, normpath
import os, sys

os.environ["CC"] = "hipcc"
os.environ["CXX"] = "hipcc"

packages = find_packages(exclude=('tests', 'doc'))
provides = ['pyhip', ]

def search_on_path(filenames):
    """Find file on system path."""
    # http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/52224

    from os.path import exists, abspath
    from os import pathsep, environ

    search_path = environ["PATH"]

    paths = search_path.split(pathsep)
    for path in paths:
        for filename in filenames:
            if exists(join(path, filename)):
                return abspath(join(path, filename))






def find_library_file(libname):
    import argparse
    """
    Try to get the directory of the specified library.
    It adds to the search path the library paths given to distutil's build_ext.
    """
    # Use a dummy argument parser to get user specified library dirs
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--library-dirs", "-L", default='')
    args, unknown = parser.parse_known_args()
    lib_dirs = args.library_dirs.split(':')
    if 'LD_LIBRARY_PATH' in os.environ:
        lib_dirs += os.environ['LD_LIBRARY_PATH'].split(':')
        # Append default search path (not a complete list)
        lib_dirs += [join(sys.prefix, 'lib'),
                '/usr/local/lib',
                '/usr/lib64',
                '/usr/lib', '/usr/lib/x86_64-linux-gnu']
        compiler = ccompiler.new_compiler()
        return compiler.find_library_file(lib_dirs,
                libname)

def find_library_dir(libname):

    lib = find_library_file(libname)
    libdirname = os.path.dirname(lib)
    base_path = os.path.abspath(libdirname)
    include_dir = os.path.abspath(os.path.join(libdirname,os.path.join('..','include')))
    return f'{libname}', libdirname,include_dir

def find_hip():
    hipcc = search_on_path(['hipcc'])
    home = os.path.abspath(os.path.join(os.path.dirname(hipcc),'..'))

    hipconfig = {'home':home, 'hipcc':hipcc,
                  'include': os.path.join(home, 'include'),
                    'lib': os.path.join(home, 'lib')}
    return hipconfig


try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

HIP = find_hip()
boost_lib = find_library_dir('boost_python')
boost_sys = find_library_dir('boost_system')
python_lib = find_library_dir('python3')

ext = Extension('pyhip._driver', sources=['src/cpp/hip.cpp','src/wrapper/wrap_hipdriv.cpp',],
                library_dirs=[boost_lib[1],python_lib[1]],
                libraries=[python_lib[0], boost_lib[0], boost_sys[0], ],
                runtime_library_dirs=[boost_sys[1],boost_lib[1],python_lib[1]],
#                extra_compile_args=['-c','-fPIC'],
                #extra_link_args =['-static'],
                include_dirs = [numpy_include,HIP['include'], python_lib[2], boost_lib[2],'src/cpp','src/wrapper'])

def customize_compiler_for_hipcc(self):

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        self.set_executable('compiler_so',HIP['hipcc'])
        super()._compile(obj, src, ext, cc_args, extra_postargs['hipcc'])
    self._compile = _compile


class custom_build_ext(build_ext):
    def build_extensions(self):
        print('RUNNING')
        self.compiler.set_executable("compiler",HIP['hipcc'])
        self.compiler.set_executable("compiler_so", HIP['hipcc']+ ' -shared')
        self.compiler.set_executable("compiler_cxx", HIP['hipcc'])
        self.compiler.set_executable("linker_so",HIP['hipcc'])
        build_ext.build_extensions(self)

#class custom_build_ext(build_ext):
#    def build_extensions(self):
#        customize_compiler_for_hip(self.compiler)
#        build_ext.build_extensions(self)

setup(name='pyhip',
        author='Ahmed Faris Al-Refaie',
        version='0.0.1',
        packages=packages,
        provides=provides,
        ext_modules = [ext],
#        cmdclass={'build_ext': custom_build_ext},
        zip_safe=False)
