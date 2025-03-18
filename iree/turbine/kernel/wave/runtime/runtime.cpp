#include <nanobind/nanobind.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/string.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <ATen/hip/HIPContext.h>
#include <unordered_map>

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
    std::string kernel, function, kernelHash;
    int sharedMemoryBytes;
    int gridX, gridY, gridZ;
    int blockX, blockY, blockZ;
};

std::unordered_map<std::string, std::tuple<hipModule_t, hipFunction_t>> cache;
using Int64Vector = std::vector<uint64_t>;
using Int32Vector = std::vector<uint32_t>;

int launch(const KernelLaunchInfo &info, const Int64Vector &tensors, const Int64Vector &dynamicDims)
{
    hipStream_t stream = at::hip::getCurrentHIPStream();
    hipModule_t module;
    hipFunction_t function;

    if (cache.count(info.kernelHash))
    {
        std::tie(module, function) = cache.at(info.kernelHash);
    }
    else
    {
        HIP_CHECK_RETURN(hipModuleLoad(&module, info.kernel.c_str()));
        HIP_CHECK_RETURN(hipModuleGetFunction(&function, module, info.function.c_str()));
        if (!info.kernelHash.empty())
            cache[info.kernelHash] = std::make_tuple(module, function);
    }

    // Since we always pass our dynamic dims as index type, iree converts them to i64
    // and then splits them to two i32s, i64 = hi | lo where
    // lo = trunc(i64) and hi = trunc(i64 >> 32).
    size_t kernArgSize = tensors.size() * sizeof(uint64_t) + 2 * dynamicDims.size() * sizeof(uint32_t);
    uint8_t kernelArguments[kernArgSize];
    uint64_t *ptr = (uint64_t *)kernelArguments;
    for (int i = 0; i < tensors.size(); i++, ptr++)
        *ptr = tensors[i];
    uint32_t *ptr2 = (uint32_t *)ptr;
    for (int i = 0; i < dynamicDims.size(); i++, ptr2++)
    {
        *ptr2 = (uint32_t)dynamicDims[i];
        ptr2++;
        *ptr2 = (uint32_t)(dynamicDims[i] >> 32);
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

    // Always unload the module if the kernel hash is empty.
    if (info.kernelHash.empty())
        HIP_CHECK_RETURN(hipModuleUnload(module));

    return hipSuccess;
}

NB_MODULE(wave_runtime, m)
{
    nb::bind_vector<Int64Vector>(m, "Int64Vector");
    nb::bind_vector<Int32Vector>(m, "Int32Vector");
    nb::class_<KernelLaunchInfo>(m, "KernelLaunchInfo")
        .def(nb::init<const std::string &, const std::string &, const std::string &, int, int, int, int, int, int, int>())
        .def_rw("kernel", &KernelLaunchInfo::kernel)
        .def_rw("function", &KernelLaunchInfo::function)
        .def_rw("kernelHash", &KernelLaunchInfo::kernelHash)
        .def_rw("sharedMemoryBytes", &KernelLaunchInfo::sharedMemoryBytes)
        .def_rw("gridX", &KernelLaunchInfo::gridX)
        .def_rw("gridY", &KernelLaunchInfo::gridY)
        .def_rw("gridZ", &KernelLaunchInfo::gridZ)
        .def_rw("blockX", &KernelLaunchInfo::blockX)
        .def_rw("blockY", &KernelLaunchInfo::blockY)
        .def_rw("blockZ", &KernelLaunchInfo::blockZ);
    m.def("launch", &launch);
}
