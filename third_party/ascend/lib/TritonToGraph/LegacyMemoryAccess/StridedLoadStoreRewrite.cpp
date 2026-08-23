/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "TritonToGraph/LegacyMemoryAccess/StridedLoadStoreRewrite.h"
#include "TritonMemoryAccess/LoadStoreMaskAnalysis.h"
#include "TritonMemoryAccess/MemoryAccessTags.h"
#include "TritonToStructured/PtrAnalysis.h"
#include "Utils/Utils.h"

#include "Dialect/TritonAscend/IR/TritonAscendDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"

#include "llvm/Support/Debug.h"

#include <cstdlib>
#include <functional>

#define DEBUG_TYPE "triton-to-linalg-indirect-load-rewrite"

namespace StridedLoadStoreRewrite {

using namespace mlir;
using namespace triton;

namespace {

// V1 fast-path supports up to 5D tensors, mirroring
// UnstructureConversionPass::tryRewriteIndirectFastPath.
constexpr size_t kFastPathRankLimit = 5;

// Returns true iff `v` is a static integer constant with |v| > 1.
static bool isStaticConstAbsGtOne(Value v) {
  IntegerAttr scalarAttr;
  if (matchPattern(v, m_Constant(&scalarAttr)))
    return std::abs(scalarAttr.getValue().getSExtValue()) > 1;
  DenseElementsAttr denseAttr;
  if (matchPattern(v, m_Constant(&denseAttr)) && denseAttr.isSplat() &&
      denseAttr.getElementType().isInteger())
    return std::abs(denseAttr.getSplatValue<llvm::APInt>().getSExtValue()) > 1;
  // Transparently see through tt.splat of a scalar constant.
  if (auto splatOp = v.getDefiningOp<triton::SplatOp>())
    return isStaticConstAbsGtOne(splatOp.getSrc());
  return false;
}

static std::optional<int64_t> getStaticConstInt(Value v) {
  IntegerAttr scalarAttr;
  if (matchPattern(v, m_Constant(&scalarAttr)))
    return scalarAttr.getValue().getSExtValue();
  DenseElementsAttr denseAttr;
  if (matchPattern(v, m_Constant(&denseAttr)) && denseAttr.isSplat() &&
      denseAttr.getElementType().isInteger())
    return denseAttr.getSplatValue<llvm::APInt>().getSExtValue();
  if (auto splatOp = v.getDefiningOp<triton::SplatOp>())
    return getStaticConstInt(splatOp.getSrc());
  return std::nullopt;
}

static std::optional<int64_t> getStaticMaskUpperBound(Value mask) {
  if (!mask)
    return std::nullopt;
  if (auto cmp = mask.getDefiningOp<arith::CmpIOp>()) {
    auto bound = getStaticConstInt(cmp.getRhs());
    if (!bound)
      return std::nullopt;
    if (cmp.getPredicate() == arith::CmpIPredicate::slt)
      return *bound;
    if (cmp.getPredicate() == arith::CmpIPredicate::sle)
      return *bound + 1;
    return std::nullopt;
  }
  if (auto andOp = mask.getDefiningOp<arith::AndIOp>()) {
    auto lhsBound = getStaticMaskUpperBound(andOp.getLhs());
    auto rhsBound = getStaticMaskUpperBound(andOp.getRhs());
    if (lhsBound && rhsBound)
      return std::min(*lhsBound, *rhsBound);
    return lhsBound ? lhsBound : rhsBound;
  }
  return std::nullopt;
}

static bool
shouldRouteMaskedSingleTilePow2ToIndirect(Value mask,
                                          RankedTensorType tensorType) {
  if (!mask || tensorType.getRank() != 1)
    return false;
  int64_t blockSize = tensorType.getShape()[0];
  if (ShapedType::isDynamic(blockSize))
    return false;
  auto upperBound = getStaticMaskUpperBound(mask);
  return upperBound && *upperBound <= blockSize;
}

// Cheaply detect tensor-level static stride > 1 before running PtrAnalysis,
// which mutates IR. Dynamic stride stays on the structured SIMD path, and
// scalar offset arithmetic does not affect per-element stride.
static bool offsetMayContainStrideGtOne(Value offset, int depthBudget = 16) {
  if (depthBudget <= 0) {
    return true; // Give up cheaply and let PtrAnalysis decide downstream.
  }
  if (!isa<RankedTensorType>(offset.getType())) {
    return false;
  }
  Operation *defOp = offset.getDefiningOp();
  if (!defOp) {
    return false;
  }
  if (auto mul = dyn_cast<arith::MulIOp>(defOp)) {
    if (isStaticConstAbsGtOne(mul.getLhs()) ||
        isStaticConstAbsGtOne(mul.getRhs())) {
      return true;
    }
    return offsetMayContainStrideGtOne(mul.getLhs(), depthBudget - 1) ||
           offsetMayContainStrideGtOne(mul.getRhs(), depthBudget - 1);
  }
  // arith.shli %a, %k effectively multiplies by 2^k; treat shift by >=1 as
  // "may contain stride > 1".
  if (auto shl = dyn_cast<arith::ShLIOp>(defOp)) {
    APInt c;
    if (matchPattern(shl.getRhs(), m_ConstantInt(&c)) &&
        c.getSExtValue() >= 1) {
      return true;
    }
    DenseElementsAttr denseAttr;
    if (matchPattern(shl.getRhs(), m_Constant(&denseAttr)) &&
        denseAttr.isSplat() && denseAttr.getElementType().isInteger() &&
        denseAttr.getSplatValue<llvm::APInt>().getSExtValue() >= 1) {
      return true;
    }
    return offsetMayContainStrideGtOne(shl.getLhs(), depthBudget - 1);
  }
  for (Value operand : defOp->getOperands()) {
    if (offsetMayContainStrideGtOne(operand, depthBudget - 1)) {
      return true;
    }
  }
  return false;
}

// Walk through shape-only wrappers to find the underlying scalar !tt.ptr<T>.
// ChunkCoalescing lifts invariant pointer tensors as
// broadcast(expand_dims(splat(ptr))), which is still a scalar base pointer for
// indirect access construction.
static Value getScalarBasePtr(Value tensorPtr, int depthBudget = 8) {
  if (depthBudget <= 0)
    return Value();
  if (auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>()) {
    Value src = splatOp.getSrc();
    if (isa<triton::PointerType>(src.getType())) {
      return src;
    }
  }
  if (auto broadcastOp = tensorPtr.getDefiningOp<triton::BroadcastOp>())
    return getScalarBasePtr(broadcastOp.getSrc(), depthBudget - 1);
  if (auto expandDimsOp = tensorPtr.getDefiningOp<triton::ExpandDimsOp>())
    return getScalarBasePtr(expandDimsOp.getSrc(), depthBudget - 1);
  return Value();
}

// Ensure the per-element offset tensor has i64 element type, matching the
// convention used elsewhere (UnstructureConversionPass::parseAddPtr).
static Value ensureI64OffsetTensor(Value offsetTensor, Location loc,
                                   PatternRewriter &rewriter) {
  auto tensorTy = dyn_cast<RankedTensorType>(offsetTensor.getType());
  if (!tensorTy)
    return Value();
  auto eltTy = dyn_cast<IntegerType>(tensorTy.getElementType());
  if (!eltTy)
    return Value();
  if (eltTy.getWidth() == 64)
    return offsetTensor;
  auto newTy =
      RankedTensorType::get(tensorTy.getShape(), rewriter.getIntegerType(64));
  return rewriter.create<arith::ExtSIOp>(loc, newTy, offsetTensor);
}

// Promote a scalar to i64. Handles i32/i64 integers and index types.
static Value ensureI64Scalar(Value v, Location loc, PatternRewriter &rewriter) {
  Type ty = v.getType();
  if (auto intTy = dyn_cast<IntegerType>(ty)) {
    if (intTy.getWidth() == 64)
      return v;
    return rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), v);
  }
  if (isa<IndexType>(ty)) {
    return rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), v);
  }
  return Value(); // Unsupported scalar type.
}

static LogicalResult unwrapScalarAddPtrChain(Value scalarPtr, Value &src,
                                             Value &scalarOffset, Location loc,
                                             PatternRewriter &rewriter) {
  src = scalarPtr;
  scalarOffset = Value();
  while (auto addPtrOp = src.getDefiningOp<triton::AddPtrOp>()) {
    if (isa<RankedTensorType>(addPtrOp.getPtr().getType()))
      break;
    if (!scalarOffset)
      scalarOffset = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI64IntegerAttr(0));
    Value offset = ensureI64Scalar(addPtrOp.getOffset(), loc, rewriter);
    if (!offset)
      return failure();
    scalarOffset = rewriter.create<arith::AddIOp>(loc, scalarOffset, offset);
    src = addPtrOp.getPtr();
  }
  return success();
}

static Value addScalarOffsetToTensor(Value offsetTensor, Value scalarOffset,
                                     Location loc, PatternRewriter &rewriter) {
  if (!scalarOffset)
    return offsetTensor;
  APInt scalarOffsetConst;
  if (matchPattern(scalarOffset, m_ConstantInt(&scalarOffsetConst)) &&
      scalarOffsetConst.isZero())
    return offsetTensor;
  auto tensorType = cast<RankedTensorType>(offsetTensor.getType());
  Value scalarOffsetTensor =
      rewriter.create<triton::SplatOp>(loc, tensorType, scalarOffset);
  return rewriter.create<arith::AddIOp>(loc, offsetTensor, scalarOffsetTensor);
}

static Value getStrideLoadOtherScalar(triton::LoadOp op,
                                      RankedTensorType resultType,
                                      PatternRewriter &rewriter) {
  auto loc = op.getLoc();
  Type elementType = resultType.getElementType();
  if (Value other = op.getOther()) {
    if (auto splatOp = other.getDefiningOp<triton::SplatOp>())
      return splatOp.getSrc();

    DenseElementsAttr denseAttr;
    if (matchPattern(other, m_Constant(&denseAttr)) && denseAttr.isSplat()) {
      if (auto floatType = dyn_cast<FloatType>(elementType)) {
        return rewriter.create<arith::ConstantOp>(
            loc, rewriter.getFloatAttr(elementType,
                                       denseAttr.getSplatValue<APFloat>()));
      }
      if (isa<IntegerType>(elementType)) {
        return rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIntegerAttr(elementType,
                                         denseAttr.getSplatValue<APInt>()));
      }
    }
    return Value();
  }

  return rewriter.create<arith::ConstantOp>(loc,
                                            rewriter.getZeroAttr(elementType));
}

static Value createStrideLoadOp(Location loc, RankedTensorType resultType,
                                Value src, Value offset, Value other,
                                ArrayRef<Value> strides, ArrayRef<Value> numels,
                                PatternRewriter &rewriter) {
  int64_t rank = resultType.getRank();
  if (static_cast<int64_t>(strides.size()) != rank ||
      static_cast<int64_t>(numels.size()) != rank) {
    return Value();
  }

  Type indexType = rewriter.getI32Type();
  auto chooseIndexType = [&](Value value) -> bool {
    Type type = value.getType();
    if (isa<IndexType>(type) || type.isInteger(64)) {
      indexType = rewriter.getI64Type();
      return true;
    }
    return type.isInteger(32);
  };
  auto castToIndexType = [&](Value value) -> Value {
    Type type = value.getType();
    if (type == indexType)
      return value;
    if (isa<IndexType>(type))
      return rewriter.create<arith::IndexCastOp>(loc, indexType, value);
    if (type.isInteger(32) && indexType.isInteger(64))
      return rewriter.create<arith::ExtSIOp>(loc, indexType, value);
    return Value();
  };
  auto chooseAll = [&](ArrayRef<Value> values) -> bool {
    for (Value value : values) {
      if (!chooseIndexType(value))
        return false;
    }
    return true;
  };

  if (!chooseIndexType(offset) || !chooseAll(strides) || !chooseAll(numels))
    return Value();
  offset = castToIndexType(offset);
  if (!offset)
    return Value();
  SmallVector<Value> castStrides;
  SmallVector<Value> castNumels;
  castStrides.reserve(rank);
  castNumels.reserve(rank);
  for (Value stride : strides) {
    stride = castToIndexType(stride);
    if (!stride)
      return Value();
    castStrides.push_back(stride);
  }
  for (Value numel : numels) {
    numel = castToIndexType(numel);
    if (!numel)
      return Value();
    castNumels.push_back(numel);
  }

  auto strideLoad = rewriter.create<triton::ascend::StrideLoadOp>(
      loc, resultType, src, offset, other, castStrides, castNumels);
  strideLoad->setAttr(RewrittenByStridedLoadStoreRewriteTAG,
                      UnitAttr::get(rewriter.getContext()));
  return strideLoad->getResult(0);
}

static Operation *createStrideStoreOp(Location loc, RankedTensorType valueType,
                                      Value dst, Value src, Value offset,
                                      ArrayRef<Value> strides,
                                      ArrayRef<Value> numels,
                                      PatternRewriter &rewriter) {
  int64_t rank = valueType.getRank();
  if (static_cast<int64_t>(strides.size()) != rank ||
      static_cast<int64_t>(numels.size()) != rank) {
    return nullptr;
  }

  Type indexType = rewriter.getI32Type();
  auto chooseIndexType = [&](Value value) -> bool {
    Type type = value.getType();
    if (isa<IndexType>(type) || type.isInteger(64)) {
      indexType = rewriter.getI64Type();
      return true;
    }
    return type.isInteger(32);
  };
  auto castToIndexType = [&](Value value) -> Value {
    Type type = value.getType();
    if (type == indexType)
      return value;
    if (isa<IndexType>(type))
      return rewriter.create<arith::IndexCastOp>(loc, indexType, value);
    if (type.isInteger(32) && indexType.isInteger(64))
      return rewriter.create<arith::ExtSIOp>(loc, indexType, value);
    return Value();
  };
  auto chooseAll = [&](ArrayRef<Value> values) -> bool {
    for (Value value : values) {
      if (!chooseIndexType(value))
        return false;
    }
    return true;
  };

  if (!chooseIndexType(offset) || !chooseAll(strides) || !chooseAll(numels))
    return nullptr;
  offset = castToIndexType(offset);
  if (!offset)
    return nullptr;
  SmallVector<Value> castStrides;
  SmallVector<Value> castNumels;
  castStrides.reserve(rank);
  castNumels.reserve(rank);
  for (Value stride : strides) {
    stride = castToIndexType(stride);
    if (!stride)
      return nullptr;
    castStrides.push_back(stride);
  }
  for (Value numel : numels) {
    numel = castToIndexType(numel);
    if (!numel)
      return nullptr;
    castNumels.push_back(numel);
  }

  auto strideStore = rewriter.create<triton::ascend::StrideStoreOp>(
      loc, dst, src, offset, castStrides, castNumels);
  strideStore->setAttr(RewrittenByStridedLoadStoreRewriteTAG,
                       UnitAttr::get(rewriter.getContext()));
  return strideStore.getOperation();
}

static Value materializeI64(OpFoldResult ofr, Location loc,
                            PatternRewriter &rewriter) {
  if (auto attr = ofr.dyn_cast<Attribute>()) {
    int64_t value = cast<IntegerAttr>(attr).getInt();
    return rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(value));
  }
  return ensureI64Scalar(llvm::cast<Value>(ofr), loc, rewriter);
}

static Value clampI64(Value value, Value lower, Value upper, Location loc,
                      PatternRewriter &rewriter) {
  value = rewriter.create<arith::MaxSIOp>(loc, value, lower);
  return rewriter.create<arith::MinSIOp>(loc, value, upper);
}

static Value getPrefixMaskNumel(Operation *op, Value mask,
                                RankedTensorType resultType,
                                PatternRewriter &rewriter) {
  if (!mask) {
    return rewriter.create<arith::ConstantOp>(
        op->getLoc(),
        rewriter.getI64IntegerAttr(resultType.getShape().front()));
  }

  auto maskState = triton::runMaskAnalysis(op, rewriter);
  if (!maskState || maskState->getRank() != 1)
    return Value();

  auto offset = getConstantIntValue(maskState->offsets.front());
  if (!offset.has_value() || offset.value() != 0)
    return Value();

  auto loc = op->getLoc();
  Value zero =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
  Value block = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(resultType.getShape().front()));
  Value numel = materializeI64(maskState->dims.front(), loc, rewriter);
  if (!numel)
    return Value();
  return clampI64(numel, zero, block, loc, rewriter);
}

static FailureOr<SmallVector<Value>>
getAddPtrStrideNumels(Operation *op, Value mask, RankedTensorType resultType,
                      PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  int64_t rank = resultType.getRank();
  ArrayRef<int64_t> shape = resultType.getShape();

  if (!mask) {
    SmallVector<Value> numels;
    for (int64_t d = 0; d < rank; ++d)
      numels.push_back(rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI64IntegerAttr(shape[d])));
    return numels;
  }

  if (rank == 1) {
    Value numel = getPrefixMaskNumel(op, mask, resultType, rewriter);
    if (!numel)
      return failure();
    return SmallVector<Value>{numel};
  }

  auto maskState = triton::runMaskAnalysis(op, rewriter);
  if (!maskState || maskState->getRank() != rank)
    return failure();

  for (int64_t d = 0; d < rank; ++d) {
    auto offsetVal = getConstantIntValue(maskState->offsets[d]);
    if (!offsetVal.has_value() || offsetVal.value() != 0)
      return failure();
  }

  Value zero =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
  SmallVector<Value> numels;
  numels.reserve(rank);
  for (int64_t d = 0; d < rank; ++d) {
    Value block = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(shape[d]));
    Value numel = materializeI64(maskState->dims[d], loc, rewriter);
    if (!numel)
      return failure();
    numel = clampI64(numel, zero, block, loc, rewriter);
    numels.push_back(numel);
  }
  return numels;
}

// AddPtr path: tt.load(tt.addptr(tt.splat(%scalar_ptr), %offsets)).
// Uses PtrAnalysis (which has IR side effects), so this function must
// stamp InspectedByStridedLoadStoreRewriteTAG on every "PtrAnalysis ran but
// don't rewrite" path -- see comment block inside.
static LogicalResult tryRewriteAddPtrLoad(triton::LoadOp op,
                                          triton::AddPtrOp addPtrOp,
                                          RankedTensorType resultType,
                                          PatternRewriter &rewriter) {
  auto loc = op.getLoc();

  // The base must resolve to a scalar pointer through shape-only wrappers.
  Value scalarBase = getScalarBasePtr(addPtrOp.getPtr());
  if (!scalarBase)
    return failure();

  // Pre-filter without mutating IR: if no per-element multiplication by a
  // constant > 1 exists in the offset chain, last stride must be 1.
  if (!offsetMayContainStrideGtOne(addPtrOp.getOffset())) {
    return failure();
  }

  // From here, PtrAnalysis may insert helper IR. Every early-out path
  // MUST stamp InspectedByStridedLoadStoreRewriteTAG and return success() so
  // the greedy driver does not re-walk the same op (which would re-run
  // PtrAnalysis and accumulate dead IR until maxIterations).
  TritonToStructured::PtrAnalysis ptrAnalysis;
  TritonToStructured::PtrState ptrState;
  auto markInspectedAndReturn = [&]() {
    op->setAttr(InspectedByStridedLoadStoreRewriteTAG,
                UnitAttr::get(rewriter.getContext()));
    return success();
  };
  if (ptrAnalysis.visitOperand(op.getPtr(), ptrState, loc, rewriter).failed())
    return markInspectedAndReturn();
  if (ptrState.stateInfo.empty())
    return markInspectedAndReturn();
  ptrState.analyzePermute();
  if (ptrState.isPermuted)
    return markInspectedAndReturn();

  // Use SIMT indirect only for static non-pow2 or masked single-tile pow2
  // strides; keep dynamic strides on the structured SIMD path.
  auto lastStrideOpt = getConstantIntValue(ptrState.stateInfo.back().stride);
  if (!lastStrideOpt.has_value())
    return markInspectedAndReturn();
  int64_t lastStride = std::abs(lastStrideOpt.value());
  if (lastStride <= 1)
    return markInspectedAndReturn();
  bool routeMaskedPow2ToIndirect =
      shouldRouteMaskedSingleTilePow2ToIndirect(op.getMask(), resultType);
  if (lastStride == 2 && !routeMaskedPow2ToIndirect)
    return markInspectedAndReturn(); // even -> deinterleave; odd -> strided DMA
  if ((lastStride & (lastStride - 1)) == 0 && !routeMaskedPow2ToIndirect)
    return markInspectedAndReturn(); // power-of-two >= 4 -> strided DMA

  bool useStrideLoad = !routeMaskedPow2ToIndirect;
  if (useStrideLoad && resultType.getRank() >= 1 && resultType.getRank() <= 3 &&
      resultType.hasStaticShape() &&
      static_cast<int64_t>(ptrState.stateInfo.size()) == resultType.getRank()) {
    Value src = ptrState.source;
    SmallVector<Value> strides;
    SmallVector<Value> numels;
    strides.reserve(resultType.getRank());

    Value baseOffset = materializeI64(ptrState.offset, loc, rewriter);
    bool operandsReady = src && baseOffset;
    for (int64_t d = 0; operandsReady && d < resultType.getRank(); ++d) {
      Value stride =
          materializeI64(ptrState.stateInfo[d].stride, loc, rewriter);
      if (!stride) {
        operandsReady = false;
        break;
      }
      strides.push_back(stride);
    }

    if (operandsReady) {
      auto numelsResult = getAddPtrStrideNumels(op.getOperation(), op.getMask(),
                                                resultType, rewriter);
      if (succeeded(numelsResult)) {
        numels = *numelsResult;
      } else {
        operandsReady = false;
      }
    }
    Value other = operandsReady
                      ? getStrideLoadOtherScalar(op, resultType, rewriter)
                      : Value();
    operandsReady = operandsReady && other;

    if (operandsReady) {
      Value strideLoadResult = createStrideLoadOp(
          loc, resultType, src, baseOffset, other, strides, numels, rewriter);
      if (!strideLoadResult)
        return markInspectedAndReturn();

      LLVM_DEBUG({
        llvm::dbgs() << "----------------------------------------------\n";
        llvm::dbgs() << "StridedLoadStoreRewrite [AddPtr]: tt.load -> "
                        "ttasc.stride_load\n";
        llvm::dbgs() << "  last_stride = " << lastStride << "\n";
        llvm::dbgs() << strideLoadResult.getDefiningOp() << "\n";
        llvm::dbgs() << "----------------------------------------------\n";
      });
      rewriter.replaceOp(op, strideLoadResult);
      return success();
    }
  }

  if (useStrideLoad && resultType.getRank() <= 3)
    return markInspectedAndReturn();

  Value offsetTensor =
      ensureI64OffsetTensor(addPtrOp.getOffset(), loc, rewriter);
  if (!offsetTensor)
    return failure();

  Value src;
  Value scalarOffset;
  if (failed(unwrapScalarAddPtrChain(scalarBase, src, scalarOffset, loc,
                                     rewriter)))
    return markInspectedAndReturn();
  offsetTensor =
      addScalarOffsetToTensor(offsetTensor, scalarOffset, loc, rewriter);

  auto indirectLoad = rewriter.create<triton::ascend::IndirectLoadOp>(
      loc, resultType, src, offsetTensor, op.getMask(), op.getOther(),
      ConverterUtils::requiresVolatileIndirectLoad(op.getPtr(), op));
  indirectLoad->setAttr(RewrittenByStridedLoadStoreRewriteTAG,
                        UnitAttr::get(rewriter.getContext()));

  LLVM_DEBUG({
    llvm::dbgs() << "----------------------------------------------\n";
    llvm::dbgs() << "StridedLoadStoreRewrite [AddPtr]: tt.load -> "
                    "tt.indirect_load\n";
    llvm::dbgs() << "  last_stride = " << lastStride << "\n";
    llvm::dbgs() << indirectLoad << "\n";
    llvm::dbgs() << "----------------------------------------------\n";
  });
  rewriter.replaceOp(op, indirectLoad.getResult());
  return success();
}

// V2 (Store) helpers ----------------------------------------------------------

// AddPtr path for tt.store. Mirrors tryRewriteAddPtrLoad but emits
// triton::ascend::StrideStoreOp when the mask can be represented by per-axis
// numel bounds.
static LogicalResult tryRewriteAddPtrStore(triton::StoreOp op,
                                           triton::AddPtrOp addPtrOp,
                                           RankedTensorType valueType,
                                           PatternRewriter &rewriter) {
  auto loc = op.getLoc();

  Value scalarBase = getScalarBasePtr(addPtrOp.getPtr());
  if (!scalarBase)
    return failure();

  if (!offsetMayContainStrideGtOne(addPtrOp.getOffset()))
    return failure();

  TritonToStructured::PtrAnalysis ptrAnalysis;
  TritonToStructured::PtrState ptrState;
  auto markInspectedAndReturn = [&]() {
    op->setAttr(InspectedByStridedLoadStoreRewriteTAG,
                UnitAttr::get(rewriter.getContext()));
    return success();
  };
  if (ptrAnalysis.visitOperand(op.getPtr(), ptrState, loc, rewriter).failed())
    return markInspectedAndReturn();
  if (ptrState.stateInfo.empty())
    return markInspectedAndReturn();
  ptrState.analyzePermute();
  if (ptrState.isPermuted)
    return markInspectedAndReturn();

  // Use SIMT indirect only for static non-pow2 or masked single-tile pow2
  // strides; keep dynamic strides on the structured SIMD path.
  auto lastStrideOpt = getConstantIntValue(ptrState.stateInfo.back().stride);
  if (!lastStrideOpt.has_value())
    return markInspectedAndReturn();
  int64_t lastStride = std::abs(lastStrideOpt.value());
  if (lastStride <= 1)
    return markInspectedAndReturn();
  bool routeMaskedPow2ToIndirect =
      shouldRouteMaskedSingleTilePow2ToIndirect(op.getMask(), valueType);
  if (lastStride == 2 && !routeMaskedPow2ToIndirect)
    return markInspectedAndReturn(); // even -> deinterleave; odd -> strided DMA
  if ((lastStride & (lastStride - 1)) == 0 && !routeMaskedPow2ToIndirect)
    return markInspectedAndReturn(); // power-of-two >= 4 -> strided DMA

  bool useStrideStore = !routeMaskedPow2ToIndirect;
  if (useStrideStore && valueType.getRank() >= 1 && valueType.getRank() <= 3 &&
      valueType.hasStaticShape() &&
      static_cast<int64_t>(ptrState.stateInfo.size()) == valueType.getRank()) {
    Value dst = ptrState.source;
    SmallVector<Value> strides;
    SmallVector<Value> numels;
    strides.reserve(valueType.getRank());

    Value baseOffset = materializeI64(ptrState.offset, loc, rewriter);
    bool operandsReady = dst && baseOffset;
    for (int64_t d = 0; operandsReady && d < valueType.getRank(); ++d) {
      Value stride =
          materializeI64(ptrState.stateInfo[d].stride, loc, rewriter);
      if (!stride) {
        operandsReady = false;
        break;
      }
      strides.push_back(stride);
    }

    if (operandsReady) {
      auto numelsResult = getAddPtrStrideNumels(op.getOperation(), op.getMask(),
                                                valueType, rewriter);
      if (succeeded(numelsResult)) {
        numels = *numelsResult;
      } else {
        operandsReady = false;
      }
    }

    if (operandsReady) {
      Operation *strideStore =
          createStrideStoreOp(loc, valueType, dst, op.getValue(), baseOffset,
                              strides, numels, rewriter);
      if (!strideStore)
        return markInspectedAndReturn();

      LLVM_DEBUG({
        llvm::dbgs() << "----------------------------------------------\n";
        llvm::dbgs() << "StridedLoadStoreRewrite [AddPtr/Store]: "
                        "tt.store -> ttasc.stride_store\n";
        llvm::dbgs() << "  last_stride = " << lastStride << "\n";
        llvm::dbgs() << *strideStore << "\n";
        llvm::dbgs() << "----------------------------------------------\n";
      });
      rewriter.eraseOp(op);
      return success();
    }
  }

  if (useStrideStore && valueType.getRank() <= 3)
    return markInspectedAndReturn();

  Value offsetTensor =
      ensureI64OffsetTensor(addPtrOp.getOffset(), loc, rewriter);
  if (!offsetTensor)
    return failure();

  Value src;
  Value scalarOffset;
  if (failed(unwrapScalarAddPtrChain(scalarBase, src, scalarOffset, loc,
                                     rewriter)))
    return markInspectedAndReturn();
  offsetTensor =
      addScalarOffsetToTensor(offsetTensor, scalarOffset, loc, rewriter);

  auto indirectStore = rewriter.create<triton::ascend::IndirectStoreOp>(
      loc, src, offsetTensor, op.getValue(), op.getMask());
  indirectStore->setAttr(RewrittenByStridedLoadStoreRewriteTAG,
                         UnitAttr::get(rewriter.getContext()));

  LLVM_DEBUG({
    llvm::dbgs() << "----------------------------------------------\n";
    llvm::dbgs() << "StridedLoadStoreRewrite [AddPtr/Store]: tt.store -> "
                    "tt.indirect_store\n";
    llvm::dbgs() << "  last_stride = " << lastStride << "\n";
    llvm::dbgs() << indirectStore << "\n";
    llvm::dbgs() << "----------------------------------------------\n";
  });
  rewriter.eraseOp(op);
  return success();
}

} // namespace

LogicalResult LoadConverter::matchAndRewrite(triton::LoadOp op,
                                             PatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  (void)loc;

  // ---- Re-entry / cross-step guards ----
  if (op->hasAttr(InspectedByStridedLoadStoreRewriteTAG))
    return failure();
  if (op->hasAttr(RewrittenByStridedLoadStoreRewriteTAG))
    return failure();
  if (op->hasAttr(mlir::triton::memory_access::ImplicitPermuteHandledTAG))
    return failure();
  if (op->hasAttr(mlir::ConverterUtils::discreteAttrName))
    return failure();

  // ---- Common early checks (AddPtr path only: upstream removed block
  // pointers from the IR, so tt.make_tensor_ptr / tt.advance can no longer
  // appear here) ----
  auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!resultType)
    return failure();
  if (resultType.getShape().size() > kFastPathRankLimit)
    return failure();

  // ---- Dispatch on source op ----
  Value ptr = op.getPtr();
  if (auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>())
    return tryRewriteAddPtrLoad(op, addPtrOp, resultType, rewriter);
  return failure();
}

LogicalResult StoreConverter::matchAndRewrite(triton::StoreOp op,
                                              PatternRewriter &rewriter) const {
  // ---- Re-entry / cross-step guards (same convention as LoadConverter) ----
  if (op->hasAttr(InspectedByStridedLoadStoreRewriteTAG))
    return failure();
  if (op->hasAttr(RewrittenByStridedLoadStoreRewriteTAG))
    return failure();
  if (op->hasAttr(mlir::triton::memory_access::ImplicitPermuteHandledTAG))
    return failure();
  if (op->hasAttr(mlir::ConverterUtils::discreteAttrName))
    return failure();

  // boundary_check was legal only on make_tensor_ptr stores, which upstream
  // removed from the IR; only the AddPtr path remains.
  auto valueType = dyn_cast<RankedTensorType>(op.getValue().getType());
  if (!valueType)
    return failure();
  if (valueType.getShape().size() > kFastPathRankLimit)
    return failure();

  Value ptr = op.getPtr();
  if (auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>())
    return tryRewriteAddPtrStore(op, addPtrOp, valueType, rewriter);
  return failure();
}

} // namespace StridedLoadStoreRewrite
