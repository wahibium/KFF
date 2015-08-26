--------------------------------------------------
      Automated GPU Kernel Transformations in
    Large-Scale Production Stencil Applications
--------------------------------------------------

author:   Mohamed Wahib (PI: Naoya Maruyama)

version:  0.1 Alpha

released: May 2015

license:  MIT License

language: C++

This project includes different components of an end-to-end framework for automatically transforming stencil-based CUDA programs to exploit inter-kernel data locality. The transformation is based on two basic operations, kernel fission and fusion, and relies on a series of steps: gathering metadata, generating graphs expressing dependencies and precedency constraints, searching for optimal kernel fissions/fusions, and code generation. Simple annotations are provided for enabling CUDA-to-CUDA transformations at which the user-written kernels are collectively replaced by auto- generated kernels optimized for locality. Driven by the flexibility required for accommodating different applications, we propose a workflow transformation approach to enable user intervention at any of the transformation steps. We demonstrate the practicality and effectiveness of automatic transformations in exploiting exposed data localities using real-world weather models of large codebases having dozens of kernels and data arrays. Experimental results show that the proposed end-to-end automated approach, with minimum intervention from the user, yields improvement in performance that is comparable to manual kernel fusion.

The project includes the following components:

1- LOGGA: a grouped genetic algorithm, which searches for optimal kernel fissions/fusions that would generate the ideal data reuse for the exposed locality.

2- Translator: a program for translating the original CUDA code to new CUDA code for which the kernel transformation was applied. The translator uses ROSE compiler to parse, change and unparse the original source code.

3- Metadata Gatherer: a set of tools to gather metadata about the performance and characteristics of the original program.

4- DDG and OEG Generators: a tools applying heuristics to extract the Data dependency Graph and Order-Execution-Graphs from the source code. The tools also allow amending the graphs.



The components mentioned above will be released in the stated order after testing and verifying each component individually. Each component is designed to be used as a standalone tool or as part of the end-to-end framework.
>>> Latest component --- LOGGA --- 


--------------------------------------------------
          Locality Optimization Grouped 
            Genetic Algorithm (LOGGA)
--------------------------------------------------


1. INTRODUCTION
----------------

The instructions for compiling and using the implementation
, version 0.1, can be found below.

The short version of the instructions for compiling the code, using
the resulting executable and some more comments follow. However, we
encourage you to read the report in order to take advantage of the
features of the implementation and understand what is actually going
on when you see all the outputs. 

2. External Dependencies
----------------

KFF depends on the following external software:

* [GNU GCC C/C++ compiler](http://gcc.gnu.org/)
* [ROSE compiler infrastructure](http://www.rosecompiler.org/)

In addition, the following platform-specific tools and libraries are
required:

* [NVIDIA CUDA toolkit and SDK for using NVIDIA GPUs](http://developer.nvidia.com/cuda-downloads) (6.5 tested)


3. COMPILATION
---------------

To compile kff, use `kff` command, which is located at
the ./bin directory, as follows:

    $ kff test.cu -I./include
    
The translated file is produced in the current working directory with its
name derived from the original file name. For example, in
the above case, a CUDA source file named `test.new.cu` will be
generated. 

For compiling LOGGA, in a separate Makefile, change the following two lines:

1) Line 35 - In the statement CC = CC, the CC on the right-hand side should be
changed to the name of a preferred C++ compiler on your machine. With
a gcc, for instance, the line should be changed to CC = gcc. 

2) Line 38 - In the statement FLAGS = -O3, the required optimization level 
should be set (for GNU gcc this is -O4, for SGI CC it is -O3). For no code 
optimization, use only OPTIMIZE = . For instance, for a maximal optimization 
with gcc, i.e. -O4, use OPTIMIZE = -O4. All modules are compiled at once since 
some compilers (as SGI CC) use intermodule optimization that does not allow
them to compile each source file separately and link them together afterwards. 

Run the following command line:

make all

After compiling the source codes, your directory should contain an
executable file; run.


4. COMMAND LINE PARAMETERS
---------------------------

There are three parameters that can be passed to the program:

<input file>      -> input file name
-h                -> help on command line parameters
-paramDescription -> print out the description of input file parameters


5. EXAMPLE INPUT FILES
-----------------------

Example input programs are located in a sub-directory examples. In "logga" diretory, files with
the names starting with input are input files and files starting with
output are output files produced with the parameters specified in the
corresponding input files. The "applications" directory include example applications used for in the end-to-end framework


5. COMMENTS
------------

This code is distributed for academic purposes with absolutely no
warranty of any kind, either expressed or implied, to the extent
permitted by applicable state law. We are not responsible for any
damage resulting from its proper or improper use.

If you have any comments or identify any bugs, please contact the
author at the following address (email is the preferred form of
communication):

Mohamed Wahib
HPC Programming Framework Research Team
RIKEN Advanced Institute for Computational Science
7-1-26, Minatojima-minami-machi, Chuo-ku
Kobe, Hyogo 650-0047
email: mohamed.attia@riken.jp
