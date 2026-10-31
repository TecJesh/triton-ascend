/* clang-format off */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>
#include <acl/acl.h>
#include <runtime/runtime/rt.h>
#include <cuda.h>

// helpers to check for acl errors
#define ACL_CHECK(ans) {{\
    npuAssert((ans), __FILE__, __LINE__);\
  }}\

static inline void npuAssert(aclError code, const char *file, int line) {{
  if (code != ACL_SUCCESS) {{
    printf("Triton Error [NPU]: %s:%d: error code 0x%x\n", file, line, code);
    exit(code);
  }}
}}

// globals
#define NPUBIN_NAME {kernel_name}_npubin
unsigned char NPUBIN_NAME[{bin_size}] = {{ {bin_data} }};
static void *{kernel_name}_func = NULL;

void load_{kernel_name}(void) {{
    if ({kernel_name}_func != NULL) return;
    ACL_CHECK(aclrtSetDevice(0));
    rtDevBinary_t devbin;
    devbin.data = (void *)&NPUBIN_NAME;
    devbin.length = {bin_size};
    devbin.magic = {magic};
    devbin.version = 0;
    void *devbinHandle = NULL;
    rtError_t rtRet = rtDevBinaryRegister(&devbin, &devbinHandle);
    if (rtRet != RT_ERROR_NONE) {{
        printf("rtDevBinaryRegister failed: 0x%x\n", rtRet);
        exit(1);
    }}
    // A single registration: one static func stub, no per-load counters.
    static size_t func_stub = 0;
    static const char *stub_name = "{register_kernel_name}_0";
    static const char *kernel_name = "{register_kernel_name}";
    rtRet = rtFunctionRegister(devbinHandle, &func_stub, stub_name, (void *)kernel_name, 0);
    if (rtRet != RT_ERROR_NONE) {{
        printf("rtFunctionRegister failed: 0x%x\n", rtRet);
        exit(1);
    }}
    {kernel_name}_func = &func_stub;
}}

void unload_{kernel_name}(void) {{
    // Device-side binary and function handles are owned by the driver;
    // there is nothing to release for NPU AOT launchers.
}}

typedef void (*triton_launch_kernel_t)(const char *name, void *function,
                                       void *stream, int gridX, int gridY,
                                       int gridZ, const int64_t *shapes,
                                       const int *ranks, int num_tensors,
                                       const int *kinds, const void *const *args,
                                       const size_t *sizes, int num_args);

/*
{kernel_docstring}
*/
CUresult {kernel_name}(CUstream stream, {signature}) {{
    if ({kernel_name}_func == NULL)
       load_{kernel_name}();
    static void *lib_handle = NULL;
    static triton_launch_kernel_t launch = NULL;
    if (launch == NULL) {{
        lib_handle = dlopen("launcher_export.so", RTLD_LAZY | RTLD_LOCAL);
        if (lib_handle == NULL) {{
            printf("Triton Error [NPU]: cannot dlopen launcher_export.so: %s\n", dlerror());
            return CUDA_ERROR_UNKNOWN;
        }}
        launch = (triton_launch_kernel_t)dlsym(lib_handle, "triton_launch_kernel");
        if (launch == NULL) {{
            printf("Triton Error [NPU]: cannot resolve triton_launch_kernel\n");
            return CUDA_ERROR_UNKNOWN;
        }}
    }}
    void *args[{num_args}] = {{ {arg_pointers} }};
    size_t arg_sizes[{num_args}] = {{ {arg_sizes} }};
    launch("{packed_kernel_name}", {kernel_name}_func, stream, {gridX}, {gridY}, {gridZ},
           NULL, NULL, 0, NULL, args, arg_sizes, {num_args});
    return CUDA_SUCCESS;
}}
