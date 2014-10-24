In experimental version will be released in November 2014
===
===
An End-To-End Automated Method for GPU Multi-Kernel Transformations to Exploit Inter-Kernel Data Locality of Stencil Applications
===

KFF is an end-to-end framework for automatically transforming stencil-based CUDA programs to exploit inter-kernel data locality. The transformation is based on two basic operations, kernel fission and fusion, and relies on a series of steps: gathering metadata, generating graphs expressing dependencies and precedency constraints, searching for optimal kernel fissions/fusions, and code generation. Simple annotations are provided for enabling CUDA-to-CUDA transformations at which the user-written kernels are collectively replaced by auto-generated kernels optimized for locality. Driven by the flexibility required for accommodating different applications, we propose a workflow transformation approach to enable user intervention at any of the transformation steps. We demonstrate the practicality and effectiveness of automatic transformations in exploiting exposed data localities using real-world weather models of large codebases having dozens of kernels and data arrays. Experimental results show that the proposed end-to-end automated approach, with minimum intervention from the user, yields improvement in performance that is comparable to manual kernel fusion.
