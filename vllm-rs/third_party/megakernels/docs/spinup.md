# Megakernels Spin-Up Document

This is a research repository to support the development of megakernels. The idea behind this project is to run an AI workload, like an LLM forward pass, on a GPU not by running dozens or hundreds of individual kernels, but by instead running one entirely fused kernel that computes every necessary step in the workload.

## High-Level Overview

### Interpreter-and-Instruction Pattern

The way that a megakernel is implemented in this repository is using an on-GPU interpreter structure. When the megakernel is launched, an interpreter is initialized on each of the GPU’s SMs that persists through the duration of the model. The interpreter iteratively reads instructions from an instruction tensor. Each instruction defines a unit of work that the SM should run, roughly analogous to the kind of work unit that a single thread block would compute when a normal kernel is launched. 

Each instruction is currently represented through 32 integers where the first integer defines the opcode. Here, each operation here loosely corresponds to the different kinds of work that we might traditionally write a separate kernel for. Examples include computing an rms norm, computing a chunk of a matmall fused with an activation function, etc.

With a normal thread block, the kernel can use its block index to figure out which chunk of the matmul output it should compute. With megakernels and instructions, the running SM should instead use the arguments inside of the instruction itself. For example, the 32 integers that we might define for a math model app might include the index of the layer we're currently computing as well as the row and column corresponding to the output tile that the instruction wants the SM to compute.

### Warp Specialization and Instruction Pipelining 

The interpreter itself benefits from heavy warp specialization. When the interpreter is launched, it contains five different kinds of warps: 
- A single controller warp, which is an administrative warp that manages things like fetching the next instruction tensor, storing out timing data, setting up and tearing down different semaphores, etc. 
- A single loader warp, responsible for fetching data. 
- Multiple (e.g. 8 or 16) consumer warps, which are responsible for doing different kinds of compute.
- A single storer warp, responsible for writing data back to GPU global memory (and sometimes performing other kinds of epilogue duties).
- A single launcher warp, which is currently unused.

Importantly, these warps can run independently of each other and are able to pipeline across instruction boundaries. The controller, which is the administrative warp that runs separately from the actual rest of the warps, works to set up the next instruction while the other warps are still computing the current instruction. Therefore, for example, if the loader warp finishes issuing its final load for an instruction, it can proceed onto the loads needed for the next instruction, even if the consumer and store still have work to do for the previous instruction.

### Virtual Memory

An important part of getting instruction pipelining to work is managing the limited amount of shared memory available on each SM. Reusing the example above, the loader wants to move on to the next instruction and start its loads. The loader needs to know which portions of shared memory are available and free to use and aren't still being used by the consumers and stores from the previous instruction.

To accomplish this, we divide the shared memory space of the SM into independently managed pages (the size of these pages is configurable, common values are 16 to 32 kilobytes). Each page, before it can be used by an instruction, must be explicitly waited upon using a special semaphore that is initialized by the controller warp. Additionally, each instruction must explicitly release each page before the instruction finishes (if an instruction doesn't need a page at all, one of its warps should just release it immediately at the beginning of the instruction). 

Each instruction declares (inside of a special user-written function) the order in which it will release shared memory pages. This allows the controller to seamlessly transition pages from one instruction to the next and enable the efficient pipelining across instruction boundaries. To do this, we introduce a distinction between logical and physical page IDs. Physical page IDs actually correspond to specific addresses in the shared memory space. However, when an instruction starts, it doesn't immediately try to use physical page 0, since that page may be held by another previous instruction, potentially for a long time. Instead, an instruction will always start by trying to use logical page 0, asks the controller for the mapping between this logical page ID and a physical page ID. The controller knows how to do this mapping because each instruction specifies the order in which it plans to release pages. For example, if the first instruction running on SM says that it's going to release page two first, the controller will then take physical page 2 and give it to the next instruction as its logical page 0.

Finally, we allocate some small region of the shared memory space to be separate from the block used to store the shared memory pages. In this special memory space, we store special values like the actual instruction data that gets loaded for every instruction. Additionally, we give each instruction a few kilobytes of scratch space in shared memory, which is often convenient for storing smaller sized values without needing to wait on or use up an entire page of shared memory.

### Writing a Megakernel

The way that we have set up this repo is with a templating system. When one wants to implement an operation, one fills out a CUDA C++ template with specific nested structs that define how this operation should use each of these different warps (except for the controller warp, which isn't directly controlled by any of the operations). For example, when writing a math model operation implementation, the loader might be responsible for running a pipeline that loads input operands from global memory. The consumers may be responsible for actually computing the matrix multiply itself, and the store may be responsible for writing out the results.

The template also defines a few special functions that are used by the controller to set up and tear down SM resources for the instruction. For example, the template must define a function that sets up any semaphores needed for inter-warp signaling and returns the number of these semaphores that are used. It also defines a function called release_lid which defines the order that this instruction will release shared memory pages in.

Generally, each one of these warp run functions takes in two important arguments. One is a globals object that contains things like pointers to all of the important buffers that are read or written to in the kernel. The other is a state object, which contains things like the actual instruction argument for the current instruction, semaphores tracking when different shared memory pages are ready, a pointer to the shared memory scratch space that's available for the current instruction, etc.

## Synchronization

One important challenge when writing megakernels is the problem of synchronization. When writing normal kernels, one can rely on the CUDA runtime to guarantee synchronization between two kernels. Since a thread block from one kernel will not run until all the thread blocks from previous kernels have finished, this makes it easy to manage data dependencies. As long as we launch our kernels in order, everything's happy. With megakernels, we don't have such guarantees. We may run into a scenario where one SM starts an instruction whose data dependencies are still running on other SMs. Because of this possibility, we must manage data dependencies ourselves. We accomplish this using a spin loop encounter system, which we also refer to as barriers in the code base.

The gist of this approach is that when we launch the mega kernel, we provide it with a tensor of integers which are originally initialized to zero. The different user-defined operations must then agree on some convention for how to appropriately index and increment these counters. For example, consider a megakernels that runs two back-to-back matrix multiplication operations. the case of a matrix multiplication operation. When the instruction finishes, the store may increment a counter whose index is a function of the layer being computed as well as the row of the output tile being computed. All columns corresponding to the same row of the output matrix will increment the same counter. 

In the following matrix multiplication operation, The user-defined implementation for the loader warps will enter a spin loop that waits for the appropriate counter to be incremented by all the necessary instructions from the previous operation.

# Correctness Testing

Since the end-to-end megakernel is often a quite complicated program, we've introduced several standards and conventions for doing incremental testing of the system.

## Python VM (aka PyVM)

A really helpful utility to create when developing a megakernel is a Python VM. This is a collection of Python functions, one for every instruction type of your megakernel, that implements the required functionality for the instruction in Python. Each pyvm function checks the appropriate barriers that have been incremented before starting the instruction, performs the appropriate computations, reading and writing from the same buffers in globals that the microkernel itself would write from, and increments the appropriate counters when the instruction finishes.

Using a PyVM allows for much more granular approaches to testing. Once a schedule of instructions has been created, we can do things like run the first and instruction types on the PyVM and then run a diff test of the megakernel by cloning the global state and then using one set of globals to run the PyVM and one set of globals to run the megakernel and looking at exactly where the differences are.