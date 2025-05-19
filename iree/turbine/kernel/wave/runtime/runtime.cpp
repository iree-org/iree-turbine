#include <nanobind/nanobind.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/string.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <ATen/hip/HIPContext.h>

namespace nb = nanobind;

#define HIP_CHECK_EXC(expr)                                                                       \
    do                                                                                            \
    {                                                                                             \
        hipError_t e = (expr);                                                                    \
        if (e)                                                                                    \
        {                                                                                         \
            const char *errName = hipGetErrorName(e);                                             \
            const char *errMsg = hipGetErrorString(e);                                            \
            std::ostringstream msg;                                                               \
            msg << "Error " << e << "(" << errName << ") " << __FILE__ << ":" << __LINE__ << ": " \
                << std::endl                                                                      \
                << #expr << std::endl                                                             \
                << errMsg << std::endl;                                                           \
            throw std::runtime_error(msg.str());                                                  \
        }                                                                                         \
    } while (0)

#define HIP_CHECK_RETURN(expr) \
    do                         \
    {                          \
        hipError_t e = (expr); \
        if (e)                 \
            return e;          \
    } while (0)

struct KernelLaunchInfo
{
    uintptr_t function; // function pointer to the kernel
    int sharedMemoryBytes;
    int gridX, gridY, gridZ;
    int blockX, blockY, blockZ;
};

using Int64Vector = std::vector<uint64_t>;
using Int32Vector = std::vector<uint32_t>;

static int launch(const KernelLaunchInfo &info, const Int64Vector &tensors,
    const Int64Vector &dynamicDims, nb::list scalarArgs)
{
    hipStream_t stream = at::hip::getCurrentHIPStream();
    hipFunction_t function = reinterpret_cast<hipFunction_t>(info.function);

    size_t scalarSize = sizeof(uint32_t); // 32-bit for both float and int

    // Since we always pass our dynamic dims as index type, iree converts them to i64
    // and then splits them to two i32s, i64 = hi | lo where
    // lo = trunc(i64) and hi = trunc(i64 >> 32).
    size_t kernArgSize = tensors.size() * sizeof(uint64_t) + 2 * dynamicDims.size() * sizeof(uint32_t) + scalarArgs.size() * scalarSize;

    uint8_t kernelArguments[kernArgSize];
    uint64_t *ptr = (uint64_t *)kernelArguments;
    for (auto val : tensors) *ptr++ = val;

    uint32_t *ptr2 = (uint32_t *)ptr;
    for (auto dim : dynamicDims) {
        *ptr2++ = static_cast<uint32_t>(dim);
        *ptr2++ = static_cast<uint32_t>(dim >> 32);
    }

    uint32_t* ptr3 = ptr2;
    // ToDo: we would like to use bit_cast in the follow-up PR.
    for (auto arg : scalarArgs){
        if (nb::isinstance<nb::int_>(arg)){
            *ptr3++ = static_cast<uint32_t>(nb::cast<uint32_t>(arg));
        }
        else if (nb::isinstance<nb::float_>(scalarArgs[0])){
            float val = nb::cast<float>(arg);
            std::memcpy(ptr3++, &val, sizeof(float));
        }
    }

    void *hipLaunchParams[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        kernelArguments,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &kernArgSize,
        HIP_LAUNCH_PARAM_END};

    HIP_CHECK_RETURN(hipModuleLaunchKernel(function, info.gridX, info.gridY, info.gridZ,
                                           info.blockX, info.blockY, info.blockZ, info.sharedMemoryBytes,
                                           stream, nullptr, (void **)&hipLaunchParams));

    return hipSuccess;
}

static void unload_binary(void *ptr) noexcept
{
    auto module = reinterpret_cast<hipModule_t>(ptr);
    if (auto e = hipModuleUnload(module)) {
        nb::print(nb::str("Failed to unload module: ") + nb::str(hipGetErrorString(e)));
    }
}

static nb::tuple load_binary(const std::string &path, const std::string &func_name)
{
    hipModule_t module;
    hipFunction_t function;
    HIP_CHECK_EXC(hipModuleLoad(&module, path.c_str()));
    HIP_CHECK_EXC(hipModuleGetFunction(&function, module, func_name.c_str()));
    nb::capsule capsule(reinterpret_cast<void *>(module), &unload_binary);
    return nb::make_tuple(capsule, reinterpret_cast<uintptr_t>(function));
}

NB_MODULE(wave_runtime, m)
{
    nb::bind_vector<Int64Vector>(m, "Int64Vector");
    nb::bind_vector<Int32Vector>(m, "Int32Vector");
    nb::class_<KernelLaunchInfo>(m, "KernelLaunchInfo")
        .def(nb::init<uintptr_t, int, int, int, int, int, int, int>())
        .def_rw("gpu_func", &KernelLaunchInfo::function)
        .def_rw("sharedMemoryBytes", &KernelLaunchInfo::sharedMemoryBytes)
        .def_rw("gridX", &KernelLaunchInfo::gridX)
        .def_rw("gridY", &KernelLaunchInfo::gridY)
        .def_rw("gridZ", &KernelLaunchInfo::gridZ)
        .def_rw("blockX", &KernelLaunchInfo::blockX)
        .def_rw("blockY", &KernelLaunchInfo::blockY)
        .def_rw("blockZ", &KernelLaunchInfo::blockZ);
    m.def("launch", &launch);
    m.def("load_binary", &load_binary);
}
