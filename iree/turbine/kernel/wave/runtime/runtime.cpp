#include <cstring>
#include <nanobind/nanobind.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/string.h>
#include <sstream>

#ifdef __linux__
#include <dlfcn.h> // dlopen
#endif

// Just hardcode necessary constants and types here, we don't expect them to
// change as it will break all hip programs in existence.
#define HIP_LAUNCH_PARAM_BUFFER_POINTER ((void*)0x01)
#define HIP_LAUNCH_PARAM_BUFFER_SIZE ((void*)0x02)
#define HIP_LAUNCH_PARAM_END ((void*)0x03)

using hipError_t = int;
using hipStream_t = void*;
using hipFunction_t = void*;
using hipModule_t = void*;

 using hipModuleLaunchKernel_t = hipError_t(*)(hipFunction_t,
    unsigned int, unsigned int,
    unsigned int, unsigned int,
    unsigned int, unsigned int,
    unsigned int, hipStream_t,
    void**, void**);

using hipGetErrorName_t = const char*(*)(hipError_t);
using hipGetErrorString_t = const char*(*)(hipError_t);
using hipModuleUnload_t = hipError_t(*)(hipModule_t);
using hipModuleLoad_t = hipError_t(*)(hipModule_t*, const char*);
using hipModuleGetFunction_t = hipError_t(*)(hipFunction_t*, hipModule_t, const char*);

static hipModuleLaunchKernel_t hipModuleLaunchKernel = nullptr;
static hipGetErrorName_t hipGetErrorName = nullptr;
static hipGetErrorString_t hipGetErrorString = nullptr;
static hipModuleUnload_t hipModuleUnload = nullptr;
static hipModuleLoad_t hipModuleLoad = nullptr;
static hipModuleGetFunction_t hipModuleGetFunction = nullptr;

static void load_hip_functions()
{
    if (hipModuleLaunchKernel && hipGetErrorName && hipGetErrorString &&
        hipModuleUnload && hipModuleLoad && hipModuleGetFunction)
        return;

#ifdef __linux__
#define GET_FUNC(module, name) do { \
    name = reinterpret_cast<decltype(name)>(dlsym(module, #name)); \
    if (!name) { \
        throw std::runtime_error("Failed to load symbol: " + std::string(#name)); \
    } \
} while (0)
    // We expect this module to be loaded permanently in the process so we don't
    // care about unloading it.
    auto module = dlopen("libamdhip64.so", RTLD_NOW);
    if (!module)
        throw std::runtime_error("Failed to load libamdhip64.so: " + std::string(dlerror()));

    GET_FUNC(module, hipModuleLaunchKernel);
    GET_FUNC(module, hipGetErrorName);
    GET_FUNC(module, hipGetErrorString);
    GET_FUNC(module, hipModuleUnload);
    GET_FUNC(module, hipModuleLoad);
    GET_FUNC(module, hipModuleGetFunction);

#undef GET_FUNC
#else
    #error "Unsupported platform"
#endif
}

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

struct KernelLaunchInfo
{
    uintptr_t stream; // hip stream pointer
    uintptr_t function; // function pointer to the kernel
    int sharedMemoryBytes;
    int gridX, gridY, gridZ;
    int blockX, blockY, blockZ;
};

using Int64Vector = std::vector<uint64_t>;
using Int32Vector = std::vector<uint32_t>;

static void launch(const KernelLaunchInfo &info, const Int64Vector &tensors,
    const Int64Vector &dynamicDims, nb::list scalarArgs)
{
    hipStream_t stream = reinterpret_cast<hipStream_t>(info.stream);
    hipFunction_t function = reinterpret_cast<hipFunction_t>(info.function);

    size_t scalarSize = sizeof(uint32_t); // 32-bit for both float and int

    // Since we always pass our dynamic dims as index type, iree converts them to i64
    // and then splits them to two i32s, i64 = hi | lo where
    // lo = trunc(i64) and hi = trunc(i64 >> 32).
    size_t kernArgSize = tensors.size() * sizeof(uint64_t) + 2 * dynamicDims.size() * sizeof(uint32_t) + scalarArgs.size() * scalarSize;

    uint8_t kernelArguments[kernArgSize];
    uint64_t *ptr = (uint64_t *)kernelArguments;
    for (auto val : tensors) *ptr++ = val;

    uint32_t* ptr2 = (uint32_t *)ptr;
    // ToDo: we would like to use bit_cast in the follow-up PR.
    for (auto arg : scalarArgs){
        if (nb::isinstance<nb::int_>(arg)){
            *ptr2++ = static_cast<uint32_t>(nb::cast<uint32_t>(arg));
        }
        else if (nb::isinstance<nb::float_>(scalarArgs[0])){
            float val = nb::cast<float>(arg);
            std::memcpy(ptr2++, &val, sizeof(float));
        }
    }

    uint32_t *ptr3 = (uint32_t *)ptr2;
    for (auto dim : dynamicDims) {
        *ptr3++ = static_cast<uint32_t>(dim);
        *ptr3++ = static_cast<uint32_t>(dim >> 32);
    }

    void *hipLaunchParams[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        kernelArguments,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &kernArgSize,
        HIP_LAUNCH_PARAM_END};

    HIP_CHECK_EXC(hipModuleLaunchKernel(function, info.gridX, info.gridY, info.gridZ,
                                        info.blockX, info.blockY, info.blockZ, 0,
                                        stream, nullptr, (void **)&hipLaunchParams));
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
        .def(nb::init<uintptr_t, uintptr_t, int, int, int, int, int, int, int>())
        .def_rw("gpu_stream", &KernelLaunchInfo::stream)
        .def_rw("gpu_func", &KernelLaunchInfo::function)
        .def_rw("sharedMemoryBytes", &KernelLaunchInfo::sharedMemoryBytes)
        .def_rw("gridX", &KernelLaunchInfo::gridX)
        .def_rw("gridY", &KernelLaunchInfo::gridY)
        .def_rw("gridZ", &KernelLaunchInfo::gridZ)
        .def_rw("blockX", &KernelLaunchInfo::blockX)
        .def_rw("blockY", &KernelLaunchInfo::blockY)
        .def_rw("blockZ", &KernelLaunchInfo::blockZ);
    m.def("load_hip_functions", &load_hip_functions);
    m.def("launch", &launch);
    m.def("load_binary", &load_binary);
}
