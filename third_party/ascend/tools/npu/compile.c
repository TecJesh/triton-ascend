/* clang-format off */
// Per-kernel AOT stub: binary registration and launcher plumbing live in
// launcher_src/launcher_cann.h, so this template only carries the
// kernel-specific state and entry points.
#define NPU_AOT_USE_ACL {use_acl_load}
#include "launcher_cann.h"

// per-kernel state and embedded binary
#define NPUBIN_NAME {kernel_name}_npubin
unsigned char NPUBIN_NAME[{bin_size}] = {{ {bin_data} }};
static size_t {kernel_name}_stub = 0;
static void *{kernel_name}_func = NULL;

void load_{kernel_name}(void) {{
    if ({kernel_name}_func != NULL) return;
    npuAotSetDevice();
    {kernel_name}_func = npuAotRegister((const unsigned char *)&NPUBIN_NAME,
                                        sizeof(NPUBIN_NAME),
                                        "{register_kernel_name}", "{register_kernel_name}_0",
                                        {mix_aiv}, &{kernel_name}_stub);
}}

void unload_{kernel_name}(void) {{
    // Device-side binary and function handles are owned by the driver;
    // there is nothing to release for NPU AOT launchers.
}}

/*
{kernel_docstring}
*/
NPUresult {kernel_name}(NPUstream stream, {signature}) {{
    if ({kernel_name}_func == NULL)
       load_{kernel_name}();
    void *args[{num_args}] = {{ {arg_pointers} }};
    size_t arg_sizes[{num_args}] = {{ {arg_sizes} }};
    return npuAotLaunch("{packed_kernel_name}", {kernel_name}_func, stream,
                        {gridX}, {gridY}, {gridZ}, args, arg_sizes, {num_args});
}}
