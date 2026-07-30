get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

include_directories(
    ${PROJECT_SOURCE_DIR}/third_party/ascend/AscendNPU-IR/bishengir/include
    ${PROJECT_BINARY_DIR}/third_party/ascend/AscendNPU-IR/bishengir/include
  )

add_llvm_executable(triton-opt triton-opt.cpp PARTIAL_SOURCES_INTENDED)

target_compile_options(triton-opt PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
target_link_libraries(triton-opt PRIVATE
  ${triton_libs}
  # Ascend backend libraries
  DiscreteMaskAccessConversion
  TritonToAnnotation
  TritonToHFusion
  TritonToHIVM
  TritonToLinalg
  TritonToLLVM
  TritonToStructured
  TritonToUnstructure
  # BiShengIR dialects
  BiShengIRAnnotationDialect
  BiShengIRHACCDialect
  BiShengIRHFusionDialect
  BiShengIRHIVMDialect
  BiShengIRMathExtDialect
  BiShengIRMemRefExtDialect
  BiShengIRScopeDialect
  BiShengIRSymbolDialect
  BiShengIRTensorDialect
  # tests
  TritonTestAnalysis
  TritonTestDialect
  TritonAMDGPUTestAnalysis
  TritonTestProton
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
)

mlir_check_all_link_libraries(triton-opt)

add_executable(triton-reduce triton-reduce.cpp)
mlir_check_all_link_libraries(triton-reduce)
target_compile_options(triton-reduce PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})

target_link_libraries(triton-reduce PRIVATE
  ${triton_libs}
  # Ascend backend libraries
  DiscreteMaskAccessConversion
  TritonToAnnotation
  TritonToHFusion
  TritonToHIVM
  TritonToLinalg
  TritonToLLVM
  TritonToStructured
  TritonToUnstructure
  # BiShengIR dialects
  BiShengIRAnnotationDialect
  BiShengIRHACCDialect
  BiShengIRHFusionDialect
  BiShengIRHIVMDialect
  BiShengIRMathExtDialect
  BiShengIRMemRefExtDialect
  BiShengIRScopeDialect
  BiShengIRSymbolDialect
  BiShengIRTensorDialect
  # tests
  TritonTestAnalysis
  TritonTestDialect
  TritonAMDGPUTestAnalysis
  TritonTestProton
  # MLIR core
  MLIRReduceLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
)

mlir_check_all_link_libraries(triton-reduce)

add_executable(triton-lsp triton-lsp.cpp)

target_compile_options(triton-lsp PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
target_link_libraries(triton-lsp PRIVATE
  ${triton_libs}
  # Ascend backend libraries
  DiscreteMaskAccessConversion
  TritonToAnnotation
  TritonToHFusion
  TritonToHIVM
  TritonToLinalg
  TritonToLLVM
  TritonToStructured
  TritonToUnstructure
  # BiShengIR dialects
  BiShengIRAnnotationDialect
  BiShengIRHACCDialect
  BiShengIRHFusionDialect
  BiShengIRHIVMDialect
  BiShengIRMathExtDialect
  BiShengIRMemRefExtDialect
  BiShengIRScopeDialect
  BiShengIRSymbolDialect
  BiShengIRTensorDialect
  # tests
  TritonTestAnalysis
  TritonTestDialect
  TritonAMDGPUTestAnalysis
  TritonTestProton
  # MLIR core
  MLIRLspServerLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
)

mlir_check_all_link_libraries(triton-lsp)


add_executable(triton-llvm-opt triton-llvm-opt.cpp)
add_dependencies(triton-llvm-opt intrinsics_gen)
target_compile_options(triton-llvm-opt PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
target_link_libraries(triton-llvm-opt PRIVATE
  TritonLLVMIR

  LLVMAnalysis
  LLVMCore
  LLVMSupport
  LLVMOption
  LLVMCodeGen
  )
export_executable_symbols_for_plugins(triton-llvm-opt)


add_executable(triton-tensor-layout triton-tensor-layout.cpp)
target_compile_options(triton-tensor-layout PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
target_link_libraries(triton-tensor-layout PRIVATE
  ${triton_libs}
  # Ascend backend libraries
  DiscreteMaskAccessConversion
  TritonToAnnotation
  TritonToHFusion
  TritonToHIVM
  TritonToLinalg
  TritonToLLVM
  TritonToStructured
  TritonToUnstructure
  # BiShengIR dialects
  BiShengIRAnnotationDialect
  BiShengIRHACCDialect
  BiShengIRHFusionDialect
  BiShengIRHIVMDialect
  BiShengIRMathExtDialect
  BiShengIRMemRefExtDialect
  BiShengIRScopeDialect
  BiShengIRSymbolDialect
  BiShengIRTensorDialect
  # tests
  TritonTestAnalysis
  TritonTestDialect
  TritonTestProton
  TritonAMDGPUTestAnalysis
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
  )

# triton-mlir-opt: Triton-Ascend optimizer combining MLIR and BishengIR dialects
add_llvm_executable(triton-mlir-opt triton-mlir-opt.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(triton-mlir-opt)

target_link_libraries(triton-mlir-opt PRIVATE
  ${triton_libs}
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
  # Core BishengIR dialect libraries
  BiShengIRAnnotationDialect
  BiShengIRHACCDialect
  BiShengIRHFusionDialect
  BiShengIRHIVMDialect
  BiShengIRMathExtDialect
  BiShengIRMemRefExtDialect
  BiShengIRScopeDialect
  BiShengIRSymbolDialect
  BiShengIRTensorDialect
)

mlir_check_all_link_libraries(triton-mlir-opt)
