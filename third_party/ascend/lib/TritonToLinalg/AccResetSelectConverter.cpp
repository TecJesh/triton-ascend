/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
//===----------------------------------------------------------------------===//
// AccResetSelectConverter
//
// Persistent matmul kernels reset their accumulator between output tiles:
//
//   %acc   = linalg.matmul ... outs(%carried) ...
//   %reset = arith.select %cond, %zeroSplat, %acc : tensor<...>
//   scf.yield ..., %reset
//
// The tensor select forces the downstream compiler to materialize the splat,
// the matmul result and the select result as distinct UB buffers, which
// breaks in-place accumulation (`remain_in_l0c`) and overflows the UB budget
// for large tiles. Rewrite it as a conditional in-place fill placed at the
// end of the block, i.e. after every other reader of %acc:
//
//   %acc = linalg.matmul ...
//   %reset = scf.if %cond {
//     %z = linalg.fill ins(%zeroScalar) outs(%acc)
//     scf.yield %z
//   } else {
//     scf.yield %acc
//   }
//   scf.yield ..., %reset
//
// Both branches yield the same underlying buffer, so no extra buffer is
// materialized. Canonicalization may then merge this scf.if into a
// same-condition side-effecting scf.if (e.g. the store epilogue), which keeps
// the reset ordered after the final store of the tile.
//===----------------------------------------------------------------------===//

#include "ascend/include/TritonToLinalg/AccResetSelectConverter.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"

#define DEBUG_TYPE "triton-to-linalg-acc-reset"

#include "llvm/Support/Debug.h"

namespace TTOpConverters {
using namespace mlir;

namespace {

/// Detects the splat side of the select and returns it together with the
/// scalar value it broadcasts. Returns {} when neither operand is a splat.
struct SplatSource {
  Value scalar;
  Operation *producer; // linalg.fill or arith.constant defining the splat
};

FailureOr<SplatSource> getSplatSource(Value operand) {
  if (auto fillOp = operand.getDefiningOp<linalg::FillOp>()) {
    if (fillOp.getInputs().size() != 1)
      return failure();
    Value input = fillOp.getInputs().front();
    if (!isa<ShapedType>(input.getType()))
      return SplatSource{input, fillOp};
    return failure();
  }
  if (auto cstOp = operand.getDefiningOp<arith::ConstantOp>()) {
    auto dense = dyn_cast<DenseElementsAttr>(cstOp.getValue());
    if (!dense || !dense.isSplat())
      return failure();
    return SplatSource{cstOp.getResult(), cstOp};
  }
  return failure();
}

} // namespace

LogicalResult
AccResetSelectConverter::matchAndRewrite(arith::SelectOp sel,
                                         PatternRewriter &rewriter) const {
  // Only tensor selects with a scalar condition are handled. Scalar selects
  // cost nothing downstream, and tensor-mask selects are masked loads, which
  // SelectCanonicalizer already lowers.
  auto tensorType = dyn_cast<RankedTensorType>(sel.getResult().getType());
  if (!tensorType || !isa<FloatType>(tensorType.getElementType()))
    return failure();
  if (isa<ShapedType>(sel.getCondition().getType()))
    return failure();

  // One side must be a splat (the reset value), the other the accumulator.
  FailureOr<SplatSource> trueSplat = getSplatSource(sel.getTrueValue());
  FailureOr<SplatSource> falseSplat = getSplatSource(sel.getFalseValue());
  bool resetIsTrue;
  SplatSource splat;
  Value acc;
  if (succeeded(trueSplat)) {
    resetIsTrue = true;
    splat = *trueSplat;
    acc = sel.getFalseValue();
  } else if (succeeded(falseSplat)) {
    resetIsTrue = false;
    splat = *falseSplat;
    acc = sel.getTrueValue();
  } else {
    return failure();
  }
  if (acc.getType() != tensorType)
    return failure();

  // The accumulator must be produced in the same block so the in-place reset
  // cannot clobber a value owned by an outer scope.
  Operation *accDef = acc.getDefiningOp();
  if (!accDef || accDef->getBlock() != sel->getBlock())
    return failure();

  Block *block = sel->getBlock();
  Operation *terminator = block->getTerminator();

  // The reset is placed just before the terminator, so every other reader of
  // %acc must live in this block, directly or in a nested region of an op
  // here; a reader in the terminator or outside the block would observe the
  // buffer after the in-place reset.
  for (Operation *user : acc.getUsers()) {
    if (user == sel.getOperation())
      continue;
    Operation *ownerInBlock = user;
    while (ownerInBlock && ownerInBlock->getBlock() != block)
      ownerInBlock = ownerInBlock->getParentOp();
    if (!ownerInBlock || ownerInBlock == terminator)
      return failure();
  }
  // The select result must only be consumed in this block so the new scf.if
  // dominates all its users.
  for (Operation *user : sel.getResult().getUsers()) {
    if (user->getBlock() != block)
      return failure();
  }

  // Build the conditional in-place reset. The splat's scalar operand
  // dominates the new scf.if: it dominates the select, which dominates the
  // insertion point.
  Value resetScalar = splat.scalar;
  rewriter.setInsertionPoint(terminator);
  auto newIf =
      rewriter.create<scf::IfOp>(sel.getLoc(), tensorType, sel.getCondition(),
                                 /*withElseRegion=*/true);
  auto emitFill = [&](OpBuilder &builder) {
    if (isa<linalg::FillOp>(splat.producer)) {
      auto fill =
          builder.create<linalg::FillOp>(sel.getLoc(), resetScalar, acc);
      return fill.getResult(0);
    }
    // Materialize the scalar splat of a tensor constant.
    auto dense = cast<DenseElementsAttr>(
        cast<arith::ConstantOp>(splat.producer).getValue());
    auto cst = builder.create<arith::ConstantOp>(
        sel.getLoc(), tensorType.getElementType(),
        dense.getSplatValue<TypedAttr>());
    auto fill =
        builder.create<linalg::FillOp>(sel.getLoc(), cst.getResult(), acc);
    return fill.getResult(0);
  };
  {
    OpBuilder::InsertionGuard guard(rewriter);
    if (resetIsTrue)
      rewriter.setInsertionPointToStart(newIf.thenBlock());
    else
      rewriter.setInsertionPointToStart(newIf.elseBlock());
    rewriter.create<scf::YieldOp>(sel.getLoc(), emitFill(rewriter));
  }
  {
    OpBuilder::InsertionGuard guard(rewriter);
    if (resetIsTrue)
      rewriter.setInsertionPointToStart(newIf.elseBlock());
    else
      rewriter.setInsertionPointToStart(newIf.thenBlock());
    rewriter.create<scf::YieldOp>(sel.getLoc(), acc);
  }

  rewriter.replaceOp(sel, newIf.getResult(0));

  // Drop the original splat when the select was its only user. The fill's
  // tensor.empty init (if any) is cleaned up by DCE.
  if (auto fillOp = dyn_cast<linalg::FillOp>(splat.producer);
      fillOp && fillOp.use_empty())
    rewriter.eraseOp(fillOp);

  LLVM_DEBUG(llvm::dbgs() << "[AccResetSelect] rewrote accumulator reset at "
                          << sel.getLoc() << "\n");
  return success();
}

} // namespace TTOpConverters
