# Project Overview and Value Proposition

Triton-Ascend is a Triton compilation framework built for the Ascend platform, aiming to enable Triton code to run efficiently on Ascend hardware.

- ## Core Value

Triton is a Python-based compilation framework that has been favored by developers in recent years. Developers only need to focus on the tile/block slicing mode and the computation logic based on tiles/blocks. During the compilation of Triton code, the compiler automatically completes memory allocation, data transfer, data computation, and pipeline parallelism based on the characteristics of underlying hardware. This greatly reduces the operator development difficulty and significantly improves the development efficiency.
Triton-Ascend adapts the Triton compilation stack to Huawei Ascend NPUs and provides a series of optimizations based on Triton, so that Triton code can run efficiently on Ascend hardware after compilation.
Currently, Triton-Ascend is still being improved. We will continuously improve the completeness of Triton Python APIs, support more data types, make memory access more flexible, and continuously optimize the automatic optimization capability of the compiler to improve the overall functionality and performance generalization of Triton-Ascend.

- ## Ascend Ecosystem Positioning

The Triton-Ascend compilation framework removes the barriers between Triton and Ascend hardware, enabling developers who are familiar with the Triton framework to use Ascend NPUs more efficiently. It provides a universal and efficient operator development paradigm, which is a key part of agile development for the Ascend software stack. This greatly enriches the Ascend operator library and upper-layer application ecosystem.

## Latest Updates and Milestones

- ### Latest Updates

Current version: [Triton-Ascend 3.2.1](https://github.com/triton-lang/triton-ascend/releases/v3.2.1)
CANN version: [CANN Community Edition 9.0.0](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0)
Version plan for 2026: Upgrade to Triton 3.5.

- #### Milestones

| Milestone  | Important Update                                                                                                                                                                                                                                                                                                                             | Status|
|------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|
| 2026.04.30 | Triton-Ascend 3.2.1 official version released                                           | ✅ |
| 2026.01.20 | Triton-Ascend 3.2.0 official version released                                                                                                                                                                                                                                                                                                | ✅ |
| 2025.11.14 | The pre-release version Triton-Ascend 3.2.0rc4 is available.<br>[Extended the tt.fp_to_fp API to support conversion to the FP8 type.](https://gitcode.com/Ascend/triton-ascend/pull/891)<br>[Added the scatter_ub_to_out API to support efficient data scattering from the UB to the GM.](https://gitcode.com/Ascend/triton-ascend/pull/864) | ✅ |
| 2025.09.30 | Improved the Triton Python APIs of the Scan/Sort class, supported non-contiguous memory access, and completed the adaptation of key Triton operators in the vLLM and sglang open-source repositories.                                                                                                                                        | ✅ |
| 2025.09.19 | Supported the extraction of the Triton-Ascend [nightly package](https://test.pypi.org/project/triton-ascend/#history).                                                                                                                                                                                                                       | ✅ |
| 2025.08.15 | Improved the support for the Triton Python APIs of the Atomic class, completed the adaptation of key Triton operators in the Flaggems open-source repository, and provided reference cases for high-performance implementation of simple operators such as Matmul.                                                                           | ✅ |
| 2025.06.30 | Supported 85% of Triton Python APIs and contiguous memory access, covering basic application scenarios.                                                                                                                                                                                                                                      | ✅ |
| 2025.05.20 | Triton-Ascend is open-source, and the GitCode code repository is alive!                                                                                                                                                                                                                                                                      | ✅ |

- #### Community Activities

1. [Meeting calendar](https://meeting.osinfra.cn/ascend)
2. [Meeting minutes dashboard](https://etherpad-ascend.meeting.osinfra.cn/p/sig-AscendNPU-IR)

## Performance

### GroupGEMM Operator Performance

We select the GroupGEMM operator as a representative example to demonstrate the performance comparison between Triton-Ascend and AscendC.

<div style="text-align: center;">
  <img src="docs/en/figures/groupgemm_speedup.svg" alt="GroupGEMM Performance" width="600">
</div>

- The Y-axis shows the speedup ratio (Speedup = AscendC_Duration_Time / Triton_Duration_Time).
- The hardware used is the `Ascend 950` series
- For operator performance tuning, please refer to the [Performance Optimization Guide](./docs/en/debug_guide/profiling.md)。
Binary wheels are available for CPython 3.10-3.14.

## Support

- ### Hardware Support

Triton-Ascend is supported by Ascend AI products. The following table lists the product models.

| Product Series                  | Product Model                             |
|----------------------------|---------------------------------------|
| **Atlas A3 training products**  | Atlas 800T A3 SuperNode server<br>Atlas 900 A3 SuperPoD server<br>A200T A3 Box8 SuperPoD server           |
| **Atlas A3 inference products**  | Atlas 800I A3 SuperNode server           |
| **Atlas A2 training products**  | Atlas 800T A2 training server<br>Atlas 900 A2 PoD cluster basic unit<br>Atlas 200T A2 Box16 heterogeneous subrack             |
| **Atlas A2 inference products**  | Atlas 800I A2 inference server<br>Atlas 300I A2 inference card<br>A200I A2 Box heterogeneous subrack             |

- #### Compatibility

**Supported OSs:**
The OSs supported by Triton-Ascend are the same as those supported by CANN. Download and install the CANN version that is compatible with your OS. For details, see the official CANN documentation.

**CANN versions:**

- Commercial versions

| Triton-Ascend Version| CANN Commercial Version | Release Date |
|-------------------|----------------------|--------------------|
| 3.2.1             | CANN 9.0.0           | 2026/04/30         |
| 3.2.0             | CANN 8.5.0           | 2026/01/16         |
| 3.2.0rc4          | CANN 8.3.RC2<br>CANN 8.3.RC1 | 2025/11/20<br>2025/10/30 |

- Community versions

| Triton-Ascend Version | CANN Community Version | Release Date |
|-------------------|----------------------|--------------------|
| 3.2.1             | CANN 9.0.0           | 2026/04/30         |
| 3.2.0             | CANN 8.5.0           | 2026/01/16         |
| 3.2.0rc4          | CANN 8.3.RC2<br>CANN 8.5.0.alpha001<br>CANN 8.3.RC1 | 2025/11/20<br>2025/11/12<br>2025/10/30 |

## Getting Started

- [Online Documentation](https://triton-ascend.readthedocs.io/zh-cn/latest/index.html)

- [Quick Start](./docs/en/quick_start.md)

- [Installation Guide](./docs/en/installation_guide.md)

- [Tutorials](./docs/en/examples/index.md)

- [Architecture Design and Core Features](./docs/en/architecture_design_and_core_features.md)

- [Operator Development Guide](./docs/en/programming_guide/index.md)

- [Operator Migration Guide](./docs/en/migration_guide/migrate_from_gpu.md)

- [Operator Debugging Guide](./docs/en/debug_guide/debugging.md#)

- [Performance Optimization Guide](./docs/en/debug_guide/profiling.md#)

- [Environment Variables and Compiler Options](docs/en/environment_variable_and_compiler_options_reference.md)

## FAQ

For details about the FAQ encountered when using Triton-Ascend, see [FAQ](./docs/en/FAQ.md#).

## Security Note

We attach great importance to the information security of developers using Triton-Ascend. For details about the security protection suggestions and related information, see [Security Note](./SECURITYNOTE.md).

## License Information

The code and documents of this project are released under the [MIT License](./LICENSE).

## Community and Contribution

You are welcome to participate in the development and code contribution of Triton-Ascend. For details, see [Contribution Guide](./CONTRIBUTING.md).

## Build Tips and Troubleshooting

- Set `TRITON_HOME=/some/path` to change the location of the `.triton`
  directory where Triton's cache is located and downloads are stored
  during the build. By default, this is the user's home directory. It
  can be changed anytime.

- If you're running out of memory when building Triton, specify the `MAX_JOBS`
  environment variable (to the `pip install -e .` command) to limit the
  number of jobs.

- Pass `--no-build-isolation` to `pip install` to make nop builds faster.
  Without this, every invocation of `pip install` uses a different symlink to
  cmake, and this forces ninja to rebuild most of the `.a` files.

- The build system creates a `compile_commands.json` file under the Triton repo
  directory. This file is used by VSCode IntelliSense and clangd to provide
  code completion and other features for C++ code.

  If IntelliSense does not work, you can try the following steps:

    - Do a local build. Run command `pip install -e .`.
    - Get the full path to the `compile_commands.json` file produced by the build:
      `find ./build -name 'compile_commands.json' | xargs readlink -f`.
      You might get a full path similar to `/Users/{username}/triton/build/cmake.macosx-11.1-arm64-cpython-3.12/compile_commands.json`.
    - In VSCode, install the
      [C/C++
      extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools),
      then open the command palette (`Shift + Command + P` on Mac, or `Shift +
      Ctrl + P` on Windows/Linux) and open `C/C++: Edit Configurations (UI)`.
    - Open "Advanced Settings" and paste the full path to
      `compile_commands.json` into the "Compile Commands" textbox.

# Running tests

There currently isn't a turnkey way to run all the Triton tests, but you can
follow the following recipe:

```shell
# One-time setup.  Note this will reinstall local Triton because torch
# overwrites it with the public version.
$ make dev-install

# To run all tests (requires a GPU)
$ make test

# Or, to run tests without a gpu
$ make test-nogpu
```

# Tips for hacking

For detailed instructions on how to debug Triton's frontend, please refer to this [tutorial](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html). The following includes additional tips for hacking on Triton's backend.

**Configuration knobs**

See [`python/triton/knobs.py`](python/triton/knobs.py) for the full list of configuration knobs. You can set those knobs directly in python or use environment variables to control them. Below are some of the environment variables you can specify (see `knobs.py` for the full list):

- `MLIR_ENABLE_DUMP=1` dumps the IR before every MLIR pass Triton runs, for all
   kernels. Use `MLIR_ENABLE_DUMP=kernelName` to dump for a specific kernel only.
  - Triton cache can interfere with the dump. In cases where `MLIR_ENABLE_DUMP=1` does not work, try cleaning your triton cache: `rm -r ~/.triton/cache/*`.
- `MLIR_DUMP_PATH` specifies where `MLIR_ENABLE_DUMP` will dump to. If unset will dump to stderr.
- `LLVM_IR_ENABLE_DUMP=1` dumps the IR before every pass run over the LLVM IR.
- `TRITON_REPRODUCER_PATH=<reproducer_path>` will generate an MLIR reproducer file
  at `<reproducer_path>` before each MLIR compiler stage. If any of the stages fail,
  `<reproducer_path>` will be a local MLIR reproducer captured right before the failing pass.
- `TRITON_INTERPRET=1` uses the Triton interpreter instead of running on the
  GPU.  You can insert Python breakpoints in your kernel code!
- `TRITON_ENABLE_LLVM_DEBUG=1` passes `-debug` to LLVM, printing a lot of
  debugging information to stdout.  If this is too noisy, run with just
  `TRITON_LLVM_DEBUG_ONLY` instead to limit the output.
  - An alternative way to reduce output noisiness is running with
  `LLVM_IR_ENABLE_DUMP=1`, extract the IR before the LLVM pass of interest, and
  then run LLVM's `opt` standalone, perhaps passing `-debug-only=foo` on the
  command line.

- `TRITON_LLVM_DEBUG_ONLY=<comma-separated>` is the equivalent of LLVM's
  `-debug-only` command-line option. This limits the LLVM debug output to
  specific pass or component names (which are specified using `#define
  DEBUG_TYPE` throughout LLVM and Triton) in order to allow the debug output to
  be less noisy. `TRITON_LLVM_DEBUG_ONLY` allows for one or more comma
  separated values to be specified (eg
  `TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions"` or
  `TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions,regalloc"`).
- `TRITON_ENABLE_ASAN=1` invokes the LLVM address sanitizer for
  memory leak and out of bounds access detection. Currently only supported on the AMD
  backend. This must be run using the ASAN libraries documented [here](https://rocm.docs.amd.com/projects/llvm-project/en/latest/conceptual/using-gpu-sanitizer.html).
  - When enabling the address sanitizer it is recommended to disable various memory caching strategies
  both within the ROCm stack and PyTorch. This will give the address sanitizer the best chance at finding the
  memory fault where it originates. See this [test](https://github.com/triton-lang/triton/blob/main/third_party/amd/python/test/test_address_sanitizer.py) for more details.

- `USE_IR_LOC={ttir,ttgir}` reparses the IR such that the location information
  will be the line number of the IR file with that particular extension,
  instead of line number of the python file. This can provide a direct mapping
  from the IR to llir/ptx. When used with performance tools, it can provide a
  breakdown on IR instructions.
- `TRITON_PRINT_AUTOTUNING=1` prints out the best autotuning config and total time
  spent for each kernel after autotuning is complete.
- `DISABLE_LLVM_OPT` will disable llvm optimizations for make_llir and make_ptx
  if its value is true when parsing as Bool. Otherwise, it will be parsed as a list
  of flags to disable llvm optimizations. One usage case is
  `DISABLE_LLVM_OPT="disable-lsr"`
  Loop strength reduction is known to cause up to 10% performance changes for
  certain kernels with register pressure.
- `TRITON_ALWAYS_COMPILE=1` forces to compile kernels regardless of cache hit.
- `MLIR_ENABLE_TIMING` dumps the timing information for each MLIR pass.
- `LLVM_ENABLE_TIMING` dumps the timing information for each LLVM pass.
- `TRITON_DEFAULT_FP_FUSION` overrides the default behavior of allowing fp fusion (mul+add->fma).
- `MLIR_ENABLE_DIAGNOSTICS=<comma-separated>` controls diagnostic emission in MLIR.
  Options are: `warnings`, `remarks`, `stacktraces`, `operations`.
  Use comma-separated values to customize output. For example,
  `MLIR_ENABLE_DIAGNOSTICS=remarks,operations` enables remarks and IR operations,
  while `MLIR_ENABLE_DIAGNOSTICS=warnings,stacktraces` enables warnings with
  stacktraces. By default, only errors are shown. Setting `warnings` includes
  errors and warnings; `remarks` includes errors, warnings, and remarks.
- `MLIR_ENABLE_REMARK` is deprecated. Please use `MLIR_ENABLE_DIAGNOSTICS=remarks`.
- `TRITON_KERNEL_DUMP` enables the dumping of the IR from each compilation stage and the final ptx/amdgcn.
- `TRITON_DUMP_DIR` specifies the directory to save the dumped IR and ptx/amdgcn when `TRITON_KERNEL_DUMP` is set to 1.
- `TRITON_KERNEL_OVERRIDE` enables the override of the compiled kernel with a user-specified IR/ptx/amdgcn at the beginning of each compilation stage.
- `TRITON_OVERRIDE_DIR` specifies the directory from which to load the IR/ptx/amdgcn files when `TRITON_KERNEL_OVERRIDE` is set to 1.
- `TRITON_F32_DEFAULT` sets the default input precision of `tl.dot` when using 32-bit floats, which can be either `ieee`, `tf32`, or `tf32x3`.
- `TRITON_FRONT_END_DEBUGGING=1` disables exception wrapping when an error occurs in the compiler frontend, allowing the full stack trace to be seen.
- `TRITON_DISABLE_LINE_INFO=1` removes all line information from the module.
- `PTXAS_OPTIONS` passes additional command-line options to the PTX assembler `ptxas` (only on NVIDIA).

> [!NOTE]
> Some of these environment variables don't have a knob in `knobs.py`-- those are only relevant to the C++ layer(s), hence they don't exist in the python layer.

**Kernel Override Steps**

```bash
export TRITON_ALWAYS_COMPILE=1
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR=<dump_dir>
export TRITON_KERNEL_OVERRIDE=1
export TRITON_OVERRIDE_DIR=<override_dir>
# Step 1: Run the kernel once to dump kernel's IRs and ptx/amdgcn in $TRITON_DUMP_DIR
# Step 2: Copy $TRITON_DUMP_DIR/<kernel_hash> to $TRITON_OVERRIDE_DIR
# Step 3: Delete the stages that you do not want to override and modify the stage you do want to override
# Step 4: Run the kernel again to see the overridden result
```

**Compiler Pipeline Inspection Steps**
To introspect the pipeline `add_stages`, before running your kernels, simply set
the add_stages_inspection_hook like so:

```python
def inspect_stages(_self, stages, options, language, capability):
    # inspect or modify add_stages here
triton.knobs.runtime.add_stages_inspection_hook = inspect_stages
```

# Changelog

Version 2.0 is out! New features include:

- Many, many bug fixes
- Performance improvements
- Backend rewritten to use MLIR
- Support for kernels that contain back-to-back matmuls (e.g., flash attention)

# Contributing

Community contributions are more than welcome, whether it be to fix bugs or to add new features at [github](https://github.com/triton-lang/triton/). For more detailed instructions, please visit our [contributor's guide](CONTRIBUTING.md).

# Compatibility

Supported Platforms:

- Linux

Supported Hardware:

- NVIDIA GPUs (Compute Capability 8.0+)
- AMD GPUs (ROCm 6.2+)
- Under development: CPUs

# Development Container (Dev Container)

**Dev Containers** for the Triton project are available from
the [triton-dev-containers repository](https://github.com/redhat-et/triton-dev-containers).

### Key Benefits:
- **Consistency**: All developers can work with the same development
  environment, ensuring uniform behavior across different systems.
- **Isolation**: The container prevents potential conflicts with software
  installed on your local machine.
- **Portability**: Easily share the development environment with team members,
  minimizing onboarding time and setup issues.

### How to Use the Dev Container:

For detailed instructions on how to use the dev containers, please see
the [dev container user guide](https://github.com/redhat-et/triton-dev-containers/blob/main/.devcontainer/devcontainer.md).
