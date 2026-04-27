#include "kittens.cuh"
#include "megakernel.cuh"
#include "pyutils/pyutils.cuh"
#include <iostream>

using namespace kittens;
using namespace megakernel;

struct h100_config
{
    // Instruction pipeline
    static constexpr int INSTRUCTION_PIPELINE_STAGES = 2;

    // num bits required to represent num pipeline stages
    static constexpr int INSTRUCTION_PIPELINE_STAGES_BITS = 1;

    static constexpr int INSTRUCTION_WIDTH = 32; // 128 bytes per instruction.
    using instruction_t = int[INSTRUCTION_WIDTH];

    // Timing info
    static constexpr int TIMING_WIDTH = 128;
    using timing_t = int[TIMING_WIDTH];

    // How many semaphores are available for dynamic use?
    static constexpr int DYNAMIC_SEMAPHORES = 32;

    // Scheduling approach -- explicit SM scheduling or work stealing?
    static constexpr bool ENABLE_GLOBAL_WORK_QUEUE = true;
    static constexpr int  GLOBAL_WORK_QUEUE_PARTITIONS = 1; // Only used if ENABLE_GLOBAL_WORK_QUEUE is true, can help minimize overhead.

    // One controller warp, one load warp, one store warp, and one mma warp.
    static constexpr int NUM_CONSUMER_WARPS = 8;
    static constexpr int NUM_WARPS = 4 + NUM_CONSUMER_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * ::kittens::WARP_THREADS;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int CLUSTER_BLOCKS = 1;
    static constexpr int MAX_SHARED_MEMORY = kittens::MAX_SHARED_MEMORY;

    // Shared memory declared statically
    static constexpr int SCRATCH_BYTES = 8192+2048;
    static constexpr int STATIC_SHARED_MEMORY = 512 + INSTRUCTION_PIPELINE_STAGES * (SCRATCH_BYTES + (INSTRUCTION_WIDTH + TIMING_WIDTH) * 4 + DYNAMIC_SEMAPHORES * 8);
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    // Shared memory declared dynamically
    static constexpr int PAGE_SIZE = 32768;
    static constexpr int NUM_PAGES = DYNAMIC_SHARED_MEMORY / PAGE_SIZE;
    static_assert(NUM_PAGES == 6, "NUM_PAGES must be 6");

    static constexpr bool TIMING_RECORD_ENABLED = false;

    static constexpr bool GMEM_SPIN_LOOP_SLEEP_NANOS = 20;

    static constexpr int CONSUMER_REGISTERS = 208;
    static constexpr int NON_CONSUMER_REGISTERS = 64;
};

struct globals_t {
    instruction_layout<h100_config> instructions;
    timing_layout<h100_config> timings;

    gl<int, 1, 1, 1, 1> global_instruction_index; // this is a global address, not a value.

    dim3 grid() { return dim3(132); }
    dim3 block() { return dim3(h100_config::NUM_THREADS); }
    int dynamic_shared_memory() { return h100_config::DYNAMIC_SHARED_MEMORY; }
};

struct test_op
{
    static constexpr int opcode = 1;

    struct controller
    {
        static __device__ int release_lid(const globals_t &g, typename h100_config::instruction_t &instruction, int &query)
        {
            return query;
        }
        static __device__ int init_semaphores(const globals_t &g, state<h100_config> &s)
        {
            return 0;
        }
    };

    struct loader
    {
        static __device__ void run(const globals_t &g, state<h100_config> &s)
        {
            warp::arrive(s.instruction_fetch_ready, h100_config::NUM_CONSUMER_WARPS);
            if(laneid() < h100_config::NUM_PAGES) {
                s.wait_page_ready(laneid());
                s.finish_page(laneid(), h100_config::NUM_CONSUMER_WARPS);
            }
        }
    };

    struct launcher
    {
        static __device__ void run(const globals_t &g, state<h100_config> &s) { }
    };
    using half_consumer = group<h100_config::NUM_CONSUMER_WARPS/2>;
    using constorer = group<h100_config::NUM_CONSUMER_WARPS + 1>;
    struct consumer
    {
        static __device__ void run(const globals_t &g, state<h100_config> &s)
        {
            if(threadIdx.x == 0) s.consumer_record(WAIT_EVENT);
            int id = s.instruction()[1];
            int sm = get_worker_id();
            if(threadIdx.x == 0) { printf("sm %d running instruction %d\n", sm, id); }
            warp::sync();
            s.consumer_record(READY_EVENT);
        }
    };

    struct storer
    {
        static __device__ void run(const globals_t &g, state<h100_config> &s) {}
    };
};

PYBIND11_MODULE(workstealing, m) {
    m.doc() = "A workstealing example";
    kittens::py::bind_kernel<
        mk<h100_config, globals_t, test_op>
    >(m, "workstealing",
        &globals_t::instructions,
        &globals_t::timings,
        &globals_t::global_instruction_index);
}