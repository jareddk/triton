﻿#include <numeric>
#include "triton/codegen/selection/generator.h"
#include "triton/codegen/target.h"
#include "triton/codegen/analysis/axes.h"
#include "triton/codegen/analysis/allocation.h"
#include "triton/codegen/analysis/align.h"
#include "triton/codegen/analysis/swizzle.h"
#include "triton/codegen/transform/coalesce.h"
#include "triton/ir/context.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/type.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

namespace triton{
namespace codegen{

using namespace llvm;

// Function for extended Euclidean Algorithm
inline int gcd_impl(int a, int b, int *x, int *y)
{
    // Base Case
    if (a == 0)
    {
        *x = 0;
        *y = 1;
        return b;
    }
    int x1, y1; // To store results of recursive call
    int gcd = gcd_impl(b%a, a, &x1, &y1);
    // Update x and y using results of
    // recursive call
    *x = y1 - (b/a) * x1;
    *y = x1;
    return gcd;
}

inline int gcd(int a, int b) {
  int x, y;
  return gcd_impl(a, b, &x, &y);
}


llvm::Instruction::BinaryOps llvm_op(ir::binary_op_t op) {
  using llop = llvm::Instruction::BinaryOps;
  using ttop = ir::binary_op_t;
  switch(op) {
    case ttop::Add: return llop::Add;
    case ttop::FAdd: return llop::FAdd;
    case ttop::Sub: return llop::Sub;
    case ttop::FSub: return llop::FSub;
    case ttop::Mul: return llop::Mul;
    case ttop::FMul: return llop::FMul;
    case ttop::UDiv: return llop::UDiv;
    case ttop::SDiv: return llop::SDiv;
    case ttop::FDiv: return llop::FDiv;
    case ttop::URem: return llop::URem;
    case ttop::SRem: return llop::SRem;
    case ttop::FRem: return llop::FRem;
    case ttop::Shl: return llop::Shl;
    case ttop::LShr: return llop::LShr;
    case ttop::AShr: return llop::AShr;
    case ttop::And: return llop::And;
    case ttop::Or: return llop::Or;
    case ttop::Xor: return llop::Xor;
  }
  throw std::runtime_error("unknown operator");
}

llvm::Instruction::CastOps llvm_op(ir::cast_op_t op) {
  using llop = llvm::Instruction::CastOps;
  using ttop = ir::cast_op_t;
  switch(op){
  case ttop::Trunc: return llop::Trunc;
  case ttop::ZExt: return llop::ZExt;
  case ttop::SExt: return llop::SExt;
  case ttop::FPTrunc: return llop::FPTrunc;
  case ttop::FPExt: return llop::FPExt;
  case ttop::UIToFP: return llop::UIToFP;
  case ttop::SIToFP: return llop::SIToFP;
  case ttop::FPToUI: return llop::FPToUI;
  case ttop::FPToSI: return llop::FPToSI;
  case ttop::PtrToInt: return llop::PtrToInt;
  case ttop::IntToPtr: return llop::IntToPtr;
  case ttop::BitCast: return llop::BitCast;
  case ttop::AddrSpaceCast: return llop::AddrSpaceCast;
  }
  throw std::runtime_error("unknown operator");
}

llvm::CmpInst::Predicate llvm_pred(ir::cmp_pred_t pred) {
  using llop = llvm::CmpInst::Predicate;
  using ttop = ir::cmp_pred_t;
  switch(pred){
    case ttop::FIRST_FCMP_PREDICATE: return llop::FIRST_FCMP_PREDICATE;
    case ttop::FCMP_FALSE: return llop::FCMP_FALSE;
    case ttop::FCMP_OEQ: return llop::FCMP_OEQ;
    case ttop::FCMP_OGT: return llop::FCMP_OGT;
    case ttop::FCMP_OGE: return llop::FCMP_OGE;
    case ttop::FCMP_OLT: return llop::FCMP_OLT;
    case ttop::FCMP_OLE: return llop::FCMP_OLE;
    case ttop::FCMP_ONE: return llop::FCMP_ONE;
    case ttop::FCMP_ORD: return llop::FCMP_ORD;
    case ttop::FCMP_UNO: return llop::FCMP_UNO;
    case ttop::FCMP_UEQ: return llop::FCMP_UEQ;
    case ttop::FCMP_UGT: return llop::FCMP_UGT;
    case ttop::FCMP_UGE: return llop::FCMP_UGE;
    case ttop::FCMP_ULT: return llop::FCMP_ULT;
    case ttop::FCMP_ULE: return llop::FCMP_ULE;
    case ttop::FCMP_UNE: return llop::FCMP_UNE;
    case ttop::FCMP_TRUE: return llop::FCMP_TRUE;
    case ttop::LAST_FCMP_PREDICATE: return llop::LAST_FCMP_PREDICATE;
    case ttop::FIRST_ICMP_PREDICATE: return llop::FIRST_ICMP_PREDICATE;
    case ttop::ICMP_EQ: return llop::ICMP_EQ;
    case ttop::ICMP_NE: return llop::ICMP_NE;
    case ttop::ICMP_UGT: return llop::ICMP_UGT;
    case ttop::ICMP_UGE: return llop::ICMP_UGE;
    case ttop::ICMP_ULT: return llop::ICMP_ULT;
    case ttop::ICMP_ULE: return llop::ICMP_ULE;
    case ttop::ICMP_SGT: return llop::ICMP_SGT;
    case ttop::ICMP_SGE: return llop::ICMP_SGE;
    case ttop::ICMP_SLT: return llop::ICMP_SLT;
    case ttop::ICMP_SLE: return llop::ICMP_SLE;
    case ttop::LAST_ICMP_PREDICATE: return llop::LAST_ICMP_PREDICATE;
  }
  throw std::runtime_error("unknown operator");
}


inline Type *llvm_type(ir::type *ty, LLVMContext &ctx) {
  // function
  if(auto* tt = dynamic_cast<ir::function_type*>(ty)){
    Type *return_ty = llvm_type(tt->get_return_ty(), ctx);
    std::vector<Type*> param_tys;
    std::transform(tt->params_begin(), tt->params_end(), std::back_inserter(param_tys),
                   [&ctx](ir::type* t){ return llvm_type(t, ctx);});
    return FunctionType::get(return_ty, param_tys, false);
  }
  // pointer
  if(ty->is_pointer_ty()){
    Type *elt_ty = llvm_type(ty->get_pointer_element_ty(), ctx);
    unsigned addr_space = ty->get_pointer_address_space();
    return PointerType::get(elt_ty, addr_space);
  }
  // integer
  if(ty->is_integer_ty()){
    unsigned bitwidth = ty->get_integer_bitwidth();
    return IntegerType::get(ctx, bitwidth);
  }
  // primitive types
  switch(ty->get_type_id()){
    case ir::type::VoidTyID:      return Type::getVoidTy(ctx);
    case ir::type::HalfTyID:      return Type::getHalfTy(ctx);
    case ir::type::FloatTyID:     return Type::getFloatTy(ctx);
    case ir::type::DoubleTyID:    return Type::getDoubleTy(ctx);
    case ir::type::X86_FP80TyID:  return Type::getX86_FP80Ty(ctx);
    case ir::type::PPC_FP128TyID: return Type::getPPC_FP128Ty(ctx);
    case ir::type::LabelTyID:     return Type::getLabelTy(ctx);
    case ir::type::MetadataTyID:  return Type::getMetadataTy(ctx);
    case ir::type::TokenTyID:     return Type::getTokenTy(ctx);
    default: break;
  }
  // unknown type
  throw std::runtime_error("unknown conversion from ir::type to Type");
}


inline llvm::Attribute llvm_attr(llvm::LLVMContext& ctx, ir::attribute attr) {
  switch(attr.get_kind()){
    case ir::noalias: return llvm::Attribute::get(ctx, llvm::Attribute::NoAlias);
    case ir::readonly: return llvm::Attribute::get(ctx, llvm::Attribute::ReadOnly);
    case ir::writeonly: return llvm::Attribute::get(ctx, llvm::Attribute::WriteOnly);
    case ir::aligned: return llvm::Attribute::get(ctx, llvm::Attribute::Alignment, attr.get_value());
    case ir::retune: return llvm::Attribute::get(ctx, llvm::Attribute::None);
    default: throw std::runtime_error("cannot convert ir::attribute_t to llvm::Attribute");
  }
}

inline bool is_trans(ir::value *v) {
  if(dynamic_cast<ir::trans_inst *>(v)) {
    return true;
  }
  if(auto *phi = dynamic_cast<ir::instruction *>(v)) {
    bool result = true;
    for(ir::value *op: phi->ops())
      result = result && is_trans(op);
    return result;
  }
  return false;
}




generator::generator(analysis::axes *a_axes,
                    analysis::layouts *layouts,
                    analysis::align *alignment,
                    analysis::allocation *alloc,
                    analysis::swizzle *swizzle,
                     target *tgt,
                    unsigned num_warps)
  : a_axes_(a_axes), layouts_(layouts), alignment_(alignment), alloc_(alloc), swizzle_(swizzle),
    tgt_(tgt), num_warps_(num_warps) {

}


void generator::visit_value(ir::value* v) {
  if(!seen_.insert(v).second)
    return;
  if(v->get_type()->is_tile_ty()){
    if(analysis::shared_layout* layout = layouts_->get(v)->to_shared()){
      auto double_buffer = layout->get_double_buffer();
      // offset
      Value *offset = nullptr;
      if(double_buffer && v == double_buffer->phi)
        offset = shared_off_[layout];
      // base pointer
      Value *ptr = shared_ptr_[layout];
      if(double_buffer && v == double_buffer->latch)
        ptr = shared_next_ptr_[layout];
      else if(double_buffer && v == double_buffer->first)
        ptr = shared_pre_ptr_[layout];
      shmems_[v] = ptr;
      shoffs_[v] = offset;
    }
  }
  // visit operands
  BasicBlock *current = builder_->GetInsertBlock();
  auto *inst = dynamic_cast<ir::instruction*>(v);
  if(inst)
    for(ir::value *op: inst->ops()){
      if(dynamic_cast<ir::constant*>(op) || !dynamic_cast<ir::phi_node*>(v))
        visit_value(op);
    }
  init_idx(v);
  // change insert point for phi node
  builder_->SetInsertPoint(current);
  auto *phi = dynamic_cast<ir::phi_node*>(v);
  if(phi && !current->empty() && current->getFirstNonPHI())
    builder_->SetInsertPoint(&*current->getFirstNonPHI());
  // visit user
  if(auto *usr = dynamic_cast<ir::user*>(v))
    usr->accept(this);
  // revert insert point
  if(phi && !current->empty() && current->getFirstNonPHI())
    builder_->SetInsertPoint(current);
}

void generator::visit_phi_node(ir::phi_node* x) {
  Type *ty = llvm_type(x->get_type()->get_scalar_ty(), *ctx_);
  for(indices_t idx: idxs_.at(x))
    vals_[x][idx] = builder_->CreatePHI(ty, x->get_num_operands());
}

void generator::visit_binary_operator(ir::binary_operator*x) {
  for(indices_t idx: idxs_.at(x)){
    Value *lhs = vals_[x->get_operand(0)][idx];
    Value *rhs = vals_[x->get_operand(1)][idx];
    vals_[x][idx] = builder_->CreateBinOp(llvm_op(x->get_op()), lhs, rhs);
  }
}

void generator::visit_getelementptr_inst(ir::getelementptr_inst* x) {
  for(indices_t idx: idxs_.at(x)){
    Value *ptr = vals_[x->get_pointer_operand()][idx];
    std::vector<Value*> vals;
    for(auto it= x->idx_begin(); it != x->idx_end(); it++)
      vals.push_back(vals_[*it][idx]);
    Type *ty = llvm_type(x->get_source_elt_ty()->get_scalar_ty(), *ctx_);
    vals_[x][idx] = builder_->CreateGEP(ty, ptr, vals);
  }
}

void generator::visit_icmp_inst(ir::icmp_inst* x) {
  for(indices_t idx: idxs_.at(x)){
    auto pred = llvm_pred(x->get_pred());
    Value *lhs = vals_[x->get_operand(0)][idx];
    Value *rhs = vals_[x->get_operand(1)][idx];
    vals_[x][idx] = builder_->CreateICmp(pred, lhs, rhs);
  }
}

void generator::visit_fcmp_inst(ir::fcmp_inst* x) {
  for(indices_t idx: idxs_.at(x)){
    auto pred = llvm_pred(x->get_pred());
    Value *lhs = vals_[x->get_operand(0)][idx];
    Value *rhs = vals_[x->get_operand(1)][idx];
    vals_[x][idx] = builder_->CreateFCmp(pred, lhs, rhs);
  }
}

void generator::visit_cast_inst(ir::cast_inst* x) {
  for(indices_t idx: idxs_.at(x)){
    Value *arg = vals_[x->get_operand(0)][idx];
    Type *ty = llvm_type(x->get_type()->get_scalar_ty(), *ctx_);
    vals_[x][idx] = builder_->CreateCast(llvm_op(x->get_op()), arg, ty);
  }
}

void generator::visit_return_inst(ir::return_inst* rr) {
  ir::value *ret_val = rr->get_return_value();
  builder_->CreateRet(ret_val ? vals_[ret_val][{}] : nullptr);
}

void generator::visit_cond_branch_inst(ir::cond_branch_inst* br) {
  BasicBlock *true_dest  = bbs_.at(br->get_true_dest());
  BasicBlock *false_dest = bbs_.at(br->get_false_dest());
  Value *cond = vals_[br->get_cond()][{}];
  builder_->CreateCondBr(cond, true_dest, false_dest);
}

void generator::visit_uncond_branch_inst(ir::uncond_branch_inst* br) {
  BasicBlock *dest = bbs_.at(br->get_dest());
  builder_->CreateBr(dest);
}


void generator::visit_unmasked_load_inst(ir::unmasked_load_inst* x) {
  throw std::runtime_error("TODO");
}

void generator::visit_masked_load_inst(ir::masked_load_inst* x) {
  // find vector size
  ir::value *_ptr = x->get_pointer_operand();
  auto order = layouts_->get(_ptr)->get_order();
  size_t ld;
  for(size_t i = 0; i < order.size(); i++){
    ld = order[i];
    if(ld < x->get_type()->get_tile_rank())
      break;
  }
  unsigned align = alignment_->get(_ptr, ld);
  unsigned vec = gcd(layouts_->get(x)->to_scanline()->nts(ld), align);
  auto idxs = idxs_.at(x);

  for(size_t i = 0; i < idxs.size(); i += vec){
    indices_t idx = idxs[i];
    Value *ptr = vals_[_ptr][idx];
    Type *ty = VectorType::get(ptr->getType()->getPointerElementType(), vec);
    int space = ptr->getType()->getPointerAddressSpace();
    ptr = builder_->CreateBitCast(ptr, PointerType::get(ty,space));
    llvm::Value* mask = vals_[x->get_mask_operand()][idx];
    PHINode *ret = builder_->CreatePHI(ptr->getType()->getPointerElementType(), 2);
    Instruction *then_term;
    Instruction *else_term;
    llvm::SplitBlockAndInsertIfThenElse(mask, ret, &then_term, &else_term);
    builder_->SetInsertPoint(then_term);
    Value* then_ret = builder_->CreateLoad(ptr);
    builder_->SetInsertPoint(else_term);
    Value *ret_false = vals_[x->get_false_value_operand()][idx];
    Value* else_ret = builder_->CreateVectorSplat(vec, ret_false);
    builder_->SetInsertPoint(ret->getParent());
    ret->addIncoming(then_ret, then_term->getParent());
    ret->addIncoming(else_ret, else_term->getParent());
    for(int ii = 0; ii < vec; ii++)
      vals_[x][idxs[i+ii]] = builder_->CreateExtractElement(ret, ii);

    //      ConstantInt *cst = nullptr;
    //      if(GetElementPtrInst *gep = dyn_cast<GetElementPtrInst>(ptr))
    //        if(gep->getNumIndices() == 1)
    //          cst = dyn_cast<ConstantInt>(gep->idx_begin());
    //      std::string offset = "";
    //      if(cst)
    //        offset = " + " + std::to_string(cst->getValue().getSExtValue()*2*vector_size);
    //      Type *fp16x2_ty = VectorType::get(builder_->getHalfTy(), 2);
    //      Type *fp16x2_pack4_ty = StructType::get(*ctx_, {fp16x2_ty, fp16x2_ty, fp16x2_ty, fp16x2_ty});
    //      FunctionType *ty = FunctionType::get(fp16x2_pack4_ty, {mask->getType(), ptr->getType()}, false);
    //      std::string asm_str;
    //      asm_str += "mov.v4.b32 {$1, $2, $3, $4}, {0, 0, 0, 0};\n";
    //      asm_str += "@$0 ld.global.nc.v4.b32 {$1, $2, $3, $4}, [$5" + offset + "];";
    //      InlineAsm *iasm = InlineAsm::get(ty, asm_str, "b,=r,=r,=r,=r,l", true);
    //      Value *current_result = builder_->CreateCall(iasm, {mask, ptr});
    //      for(unsigned i = 0; i < vector_size; i++){
    //        Value *tmp = builder_->CreateExtractValue(current_result, {i / 2});
    //        Value *v = builder_->CreateExtractElement(tmp, i % 2);
    //        result->set_value(result->get_ordered_indices(linear + i), v);
    //      }
  }
}

void generator::visit_unmasked_store_inst(ir::unmasked_store_inst* st) {
  throw std::runtime_error("TODO");
}



void generator::visit_masked_store_inst(ir::masked_store_inst* x) {
  // vector size
  int vec = 1;
  int ld = layouts_->get(x->get_pointer_operand())->get_order()[0];
  unsigned align = alignment_->get(x->get_pointer_operand(), ld);
  vec = gcd(layouts_->get(x->get_pointer_operand())->to_scanline()->nts(ld), align);
  //
  auto idxs = idxs_.at(x->get_value_operand());
  for(size_t i = 0; i < idxs.size(); i += vec){
    auto idx = idxs[i];
    Value* elt = UndefValue::get(VectorType::get(vals_[x->get_value_operand()][idx]->getType(), vec));
    for(int ii = 0; ii < vec; ii++)
      elt = builder_->CreateInsertElement(elt, vals_[x->get_value_operand()][idxs[i+ii]], ii);
    Value *ptr = vals_[x->get_pointer_operand()][idx];
    Value *pred = vals_[x->get_mask_operand()][idx];
    // type information
    Type *ty = elt->getType();
    unsigned nbits = ty->getScalarSizeInBits();
    unsigned subword_pack = 1;
    int supervec_size = vec;
    if(nbits < 32 && vec >= 2){
      subword_pack = 32 / nbits;
      supervec_size /= subword_pack;
      nbits = 32;
    }
    unsigned nbytes = nbits / 8;
    // extract pointer offset
    std::string offset = "";
    if(GetElementPtrInst *gep = dyn_cast<GetElementPtrInst>(ptr))
    if(gep->getNumIndices() == 1)
    if(ConstantInt *cst = dyn_cast<ConstantInt>(gep->idx_begin())){
      offset = " + " + std::to_string(cst->getValue().getSExtValue()*nbytes);
      ptr = gep->getPointerOperand();
    }
    ptr = builder_->CreateBitCast(ptr, ty->getPointerTo(1));
    // asm argument type
    std::vector<Type*> arg_ty = {pred->getType(), ptr->getType()};
    for(int v = 0; v < supervec_size; v++){
      if(subword_pack == 1)
        arg_ty.push_back(ty->getScalarType());
      else
        arg_ty.push_back(VectorType::get(ty->getScalarType(), subword_pack));
    }
    // asm function type
    FunctionType *fn_ty = FunctionType::get(builder_->getVoidTy(), arg_ty, false);
    // asm string
    std::string asm_str;
    asm_str += "@$0 st.global";
    if(supervec_size > 1)
      asm_str += ".v" + std::to_string(supervec_size);
    asm_str += ".b" + std::to_string(nbits) + " [$1" + offset + "],";
    if(supervec_size > 1)
      asm_str += "{";
    for(int v = 0; v < supervec_size; v++){
      if(v > 0)
        asm_str += ", ";
      asm_str += "$" + std::to_string(2 + v);
    }
    if(supervec_size > 1)
      asm_str += "}";
    asm_str += ";";
    // asm constraint
    std::string constraint = "b,l";
    for(int v = 0; v < supervec_size; v++){
      constraint += ",";
      constraint += (nbits == 32 ? "r" : "h");
    }
    // create inline asm
    InlineAsm *iasm = InlineAsm::get(fn_ty, asm_str, constraint, true);
    // call asm
    std::vector<Value*> args = {pred, ptr};
    for(int v = 0; v < supervec_size; v++){
      Value* curr;
      if(subword_pack == 1)
        curr = builder_->CreateExtractElement(elt, builder_->getInt32(v));
      else {
        curr = UndefValue::get(VectorType::get(ty->getScalarType(), subword_pack));
        for(int i = 0; i < subword_pack; i++)
          curr = builder_->CreateInsertElement(curr, builder_->CreateExtractElement(elt, builder_->getInt32(v*subword_pack + i)), builder_->getInt32(i));
      }
      args.push_back(curr);
    }
    builder_->CreateCall(iasm, args);
  }
}


void generator::visit_reshape_inst(ir::reshape_inst* x) {
  auto idxs = idxs_.at(x);
  for(size_t i = 0; i < idxs_.at(x).size(); i ++){
    ir::value* op = x->get_operand(0);
    vals_[x][idxs_[x][i]] = vals_[op][idxs_[op][i]];
  };
}

void generator::visit_splat_inst(ir::splat_inst* x) {
  for(auto idx: idxs_.at(x)){
    vals_[x][idx] = vals_[x->get_operand(0)][{}];
  }
}

void generator::visit_broadcast_inst(ir::broadcast_inst* x) {
  ir::value* in = x->get_operand(0);
  const auto& in_shapes = in->get_type()->get_tile_shapes();
  for(auto out_idx: idxs_.at(x)){
    indices_t in_idx = out_idx;
    for(size_t k = 0; k < in_idx.size(); k++)
      in_idx[k] = in_shapes[k] == 1 ? builder_->getInt32(0) : in_idx[k];
    vals_[x][out_idx] = vals_[in][in_idx];
  }
}

void generator::visit_downcast_inst(ir::downcast_inst* x) {
  vals_[x][{}] = vals_[x->get_operand(0)][{builder_->getInt32(0)}];
}

void generator::visit_get_program_id_inst(ir::get_program_id_inst* pid) {
  Module *module = builder_->GetInsertBlock()->getModule();
  Value *ret = tgt_->get_block_id(module, *builder_, pid->get_axis());
  vals_[pid][{}] = ret;
}

void generator::visit_get_num_program_inst(ir::get_num_program_inst* np) {
  Module *module = builder_->GetInsertBlock()->getModule();
  Value *ret = tgt_->get_num_blocks(module, *builder_, np->get_axis());
  vals_[np][{}] = ret;
}

void generator::visit_exp_inst(ir::exp_inst* x){
//  Function *fn = builder_->GetInsertBlock()->getParent();
//  Module *module = fn->getParent();
//  Type *ty = llvm_type(x->get_type()->get_scalar_ty(), *ctx_);
//  Function *ex2 = Intrinsic::getDeclaration(module, Intrinsic::nvvm_ex2_approx_ftz_f, {ty});
  Constant *log2e = ConstantFP::get(builder_->getFloatTy(), 1.4426950408889634);
  std::vector<llvm::Type*> tys = {builder_->getFloatTy()};
  FunctionType *fn_ty = FunctionType::get(builder_->getFloatTy(), tys, false);
  InlineAsm *ex2 = InlineAsm::get(fn_ty, "ex2.approx.f32 $0, $1;", "=f,f", false);
  for(auto idx: idxs_.at(x)){
    Value *ex2arg = builder_->CreateFMul(vals_[x->get_operand(0)][idx], log2e);
    vals_[x][idx] = builder_->CreateCall(ex2, std::vector<llvm::Value*>{ex2arg});
  }
}

void generator::visit_log_inst(ir::log_inst* x){
//  Function *fn = builder_->GetInsertBlock()->getParent();
//  Module *module = fn->getParent();
//  Type *ty = llvm_type(x->get_type()->get_scalar_ty(), *ctx_);
//  Function *ex2 = Intrinsic::getDeclaration(module, Intrinsic::nvvm_ex2_approx_ftz_f, {ty});
  Constant *rcplog2e = ConstantFP::get(builder_->getFloatTy(), 0.6931471805599453);
  std::vector<llvm::Type*> tys = {builder_->getFloatTy()};
  FunctionType *fn_ty = FunctionType::get(builder_->getFloatTy(), tys, false);
  InlineAsm *lg2 = InlineAsm::get(fn_ty, "lg2.approx.f32 $0, $1;", "=f,f", false);
  for(auto idx: idxs_.at(x)){
    Value *lg2arg = builder_->CreateCall(lg2, std::vector<llvm::Value*>{vals_[x->get_operand(0)][idx]});
    vals_[x][idx] = builder_->CreateFMul(lg2arg, rcplog2e);
  }
}

void generator::visit_atomic_cas_inst(ir::atomic_cas_inst* cas) {
  BasicBlock *current = builder_->GetInsertBlock();
  Module *module = current->getModule();
  Value *tid = tgt_->get_local_id(module, *builder_, 0);
  Value *pred = builder_->CreateICmpEQ(tid, builder_->getInt32(0));
  BasicBlock *tid_0_bb = BasicBlock::Create(*ctx_, "tid_0", current->getParent());
  BasicBlock *tid_0_done_bb = BasicBlock::Create(*ctx_, "tid_0_done", current->getParent());
  tgt_->add_barrier(module, *builder_);
  tgt_->add_memfence(module, *builder_);
  builder_->CreateCondBr(pred, tid_0_bb, tid_0_done_bb);
  builder_->SetInsertPoint(tid_0_bb);
  Value *cas_ptr = vals_[cas->get_operand(0)][{}];
  Value *cas_cmp = vals_[cas->get_operand(1)][{}];
  Value *cas_val = vals_[cas->get_operand(2)][{}];
  Value *old = builder_->CreateAtomicCmpXchg(cas_ptr, cas_cmp, cas_val,
                                             AtomicOrdering::Monotonic,
                                             AtomicOrdering::Monotonic);
  old = builder_->CreateExtractValue(old, std::vector<unsigned>{0});
  Value *atom_ptr;
  atom_ptr = builder_->CreateGEP(shmem_, builder_->getInt32(alloc_->offset(layouts_->get(layouts_->tmp(cas)))));
  atom_ptr = builder_->CreateBitCast(atom_ptr, PointerType::get(old->getType(), 3));

  builder_->CreateStore(old, atom_ptr);
  builder_->CreateBr(tid_0_done_bb);
  builder_->SetInsertPoint(tid_0_done_bb);
  tgt_->add_memfence(module, *builder_);
  tgt_->add_barrier(module, *builder_);
  vals_[cas][{}] = builder_->CreateLoad(atom_ptr);
}

void generator::visit_atomic_exch_inst(ir::atomic_exch_inst* xchg) {
  BasicBlock *current = builder_->GetInsertBlock();
  Module *module = current->getModule();
  Value *rmw_ptr = vals_[xchg->get_operand(0)][{}];
  Value *rmw_val = vals_[xchg->get_operand(1)][{}];
  Value *tid = tgt_->get_local_id(module, *builder_, 0);
  Value *pred = builder_->CreateICmpEQ(tid, builder_->getInt32(0));
  BasicBlock *tid_0_bb = BasicBlock::Create(*ctx_, "tid_0", current->getParent());
  BasicBlock *tid_0_done_bb = BasicBlock::Create(*ctx_, "tid_0_done", current->getParent());
  tgt_->add_memfence(module, *builder_);
  tgt_->add_barrier(module, *builder_);
  builder_->CreateCondBr(pred, tid_0_bb, tid_0_done_bb);
  builder_->SetInsertPoint(tid_0_bb);
  builder_->CreateAtomicRMW(AtomicRMWInst::Xchg, rmw_ptr, rmw_val,
                                          AtomicOrdering::Monotonic,
                                          SyncScope::System);
  builder_->CreateBr(tid_0_done_bb);
  builder_->SetInsertPoint(tid_0_done_bb);
  tgt_->add_memfence(module, *builder_);
}

void generator::visit_atomic_add_inst(ir::atomic_add_inst* add) {


  if(add->get_type()->is_tile_ty()){
    ir::value* ptr = add->get_operand(0);
    ir::value* val = add->get_operand(1);
    ir::value* msk = add->get_operand(2);

    // vector size
    int vec = 1;
    int ld = layouts_->get(ptr)->get_order()[0];
    unsigned alignment = alignment_->get(ptr, ld);
    vec = gcd(layouts_->get(ptr)->to_scanline()->nts(ld), alignment);
    vec = std::min(vec, val->get_type()->get_tile_element_ty()->is_half_ty() ? 2 : 1);

    for(int i = 0; i < idxs_.at(val).size(); i += vec){
      auto idx = idxs_[val][i];
      Value *rmw_val = UndefValue::get(VectorType::get(vals_[val][idx]->getType(), vec));
      for(int ii = 0; ii < vec; ii++)
        rmw_val = builder_->CreateInsertElement(rmw_val, vals_[val][idxs_[val][i+ii]], ii);
      Value *rmw_ptr = vals_[ptr][idx];
      Value *rmw_msk = vals_[msk][idx];
      if(vec == 1)
        rmw_val = builder_->CreateExtractElement(rmw_val, builder_->getInt32(0));
      Type* ty = rmw_val->getType();
      size_t nbits = ty->getScalarSizeInBits();
      // extract pointer offset
      std::string offset = "";
      if(GetElementPtrInst *gep = dyn_cast<GetElementPtrInst>(rmw_ptr))
      if(gep->getNumIndices() == 1)
      if(ConstantInt *cst = dyn_cast<ConstantInt>(gep->idx_begin())){
        offset = " + " + std::to_string(cst->getValue().getSExtValue()*nbits/8);
        rmw_ptr = gep->getPointerOperand();
      }
      rmw_ptr = builder_->CreateBitCast(rmw_ptr, ty->getPointerTo(1));
      // asm argument type
      std::vector<Type*> arg_ty = {rmw_msk->getType(), rmw_ptr->getType(), rmw_val->getType()};
      // asm function type
      FunctionType *fn_ty = FunctionType::get(ty, arg_ty, false);
      // asm string
      std::string suffix = vec == 2 ? "x2" : "";
      std::string mod = nbits == 32 ? "" : ".noftz";
      std::string asm_str = "@$0 atom.global.gpu.add" + mod + ".f" + std::to_string(nbits) + suffix + " $1, [$2" + offset + "], $3;";
      std::string ty_id = nbits == 32 ? "f" : (vec == 1 ? "h" : "r");
      std::string constraint = "b,=" + ty_id + ",l," + ty_id;
      // create inline asm
      InlineAsm *iasm = InlineAsm::get(fn_ty, asm_str, constraint, true);
      // call asm
      builder_->CreateCall(iasm, {rmw_msk, rmw_ptr, rmw_val});
    }
  }
  else{
    Value *rmw_ptr = vals_[add->get_operand(0)][{}];
    Value *rmw_val = vals_[add->get_operand(1)][{}];
    Value *rmw_msk = vals_[add->get_operand(2)][{}];
    Type* ty = rmw_val->getType();
    size_t nbits = ty->getScalarSizeInBits();
    std::vector<Type*> arg_ty = {rmw_msk->getType(), rmw_ptr->getType(), rmw_val->getType()};
    FunctionType *fn_ty = FunctionType::get(ty, arg_ty, false);
    std::string mod = nbits == 32 ? "" : ".noftz";
    std::string asm_str = "@$0 atom.global.gpu.add" + mod + ".f" + std::to_string(nbits) + " $1, [$2], $3;";
    std::string ty_id = nbits == 32 ? "f" : "h";
    InlineAsm *iasm = InlineAsm::get(fn_ty, asm_str, "b,="+ty_id+",l,"+ty_id, true);

    BasicBlock *current = builder_->GetInsertBlock();
    Module *module = current->getModule();

    Value *tid = tgt_->get_local_id(module, *builder_, 0);
    Value *pred = builder_->CreateICmpEQ(tid, builder_->getInt32(0));
    BasicBlock *tid_0_bb = BasicBlock::Create(*ctx_, "tid_0", current->getParent());
    BasicBlock *tid_0_done_bb = BasicBlock::Create(*ctx_, "tid_0_done", current->getParent());
    tgt_->add_memfence(module, *builder_);
    tgt_->add_barrier(module, *builder_);
    builder_->CreateCondBr(pred, tid_0_bb, tid_0_done_bb);
    builder_->SetInsertPoint(tid_0_bb);
    builder_->CreateCall(iasm, {rmw_msk, rmw_ptr, rmw_val});
    builder_->CreateBr(tid_0_done_bb);
    builder_->SetInsertPoint(tid_0_done_bb);
    tgt_->add_memfence(module, *builder_);
  }
}

void generator::visit_hmma_dot(ir::dot_inst* dot, ir::value *A, ir::value *B, ir::value *D, unsigned NK) {
  const auto& shapes = dot->get_type()->get_tile_shapes();

  std::map<std::vector<Value*>, std::vector<Value*>> fcs;

  for(indices_t idx: idxs_.at(dot)){
    std::vector<Value*> key(idx.size() - 2);
    std::copy(idx.begin() + 2, idx.end(), key.begin());
    fcs[key].push_back(vals_[D][idx]);
  };

  auto shape_a = A->get_type()->get_tile_shapes();
  auto shape_b = B->get_type()->get_tile_shapes();
  auto ord_a = layouts_->get(A)->get_order();
  auto ord_b = layouts_->get(B)->get_order();

  if(tgt_->as_nvidia()->sm() < 80){

    analysis::mma_layout* layout = layouts_->get(dot)->to_mma884();
    analysis::shared_layout* layout_a = (analysis::shared_layout*)layouts_->get(dot->get_operand(0));
    analysis::shared_layout* layout_b = (analysis::shared_layout*)layouts_->get(dot->get_operand(1));

    size_t a_vec = swizzle_->get_vec(layout_a);
    size_t b_vec = swizzle_->get_vec(layout_b);
    Value* _a_vec = builder_->getInt32(a_vec);
    Value* _b_vec = builder_->getInt32(b_vec);

    bool is_a_row = ord_a[0] != 0;
    bool is_b_row = ord_b[0] != 0;

    int stride_am = is_a_row ? shape_a[1] : 1;
    int stride_ak = is_a_row ? 1 : shape_a[0];
    int stride_bn = is_b_row ? 1 : shape_b[0];
    int stride_bk = is_b_row ? shape_b[1] : 1;

    unsigned stride_rep_m = layout->wpt(0) * layout->fpw(0) * 8;
    unsigned stride_rep_n = layout->wpt(1) * layout->fpw(1) * 8;
    unsigned stride_rep_k = 1;
    unsigned num_rep_i = shapes[0] / stride_rep_m;
    unsigned ld_fc = num_rep_i * 2;

    Type *fp32_ty = builder_->getFloatTy();
    Type *fp16x2_ty = VectorType::get(builder_->getHalfTy(), 2);
    Type *fp32_pack8_ty = StructType::get(*ctx_, std::vector<llvm::Type*>{fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty});
    FunctionType *mma_ty = FunctionType::get(fp32_pack8_ty, std::vector<llvm::Type*>{fp16x2_ty, fp16x2_ty, fp16x2_ty, fp16x2_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty}, false);
    std::string op_a = is_a_row ? "row" : "col";
    std::string op_b = is_b_row ? "row" : "col";
    InlineAsm *mma_fn = InlineAsm::get(mma_ty, " mma.sync.aligned.m8n8k4." + op_a + "." + op_b + ".f32.f16.f16.f32 "
                                               "{$0, $1, $2, $3, $4, $5, $6, $7}, "
                                               "{$8, $9}, "
                                               "{$10, $11}, "
                                               "{$0, $1, $2, $3, $4, $5, $6, $7};", "=f,=f,=f,=f,=f,=f,=f,=f,r,r,r,r,0,1,2,3,4,5,6,7", false);


    BasicBlock* CurrBB = builder_->GetInsertBlock();
    BasicBlock* FirstBB = &CurrBB->getParent()->getEntryBlock();
    builder_->SetInsertPoint(FirstBB->getTerminator());

    int per_phase_a = swizzle_->get_per_phase(layout_a);
    int max_phase_a = swizzle_->get_max_phase(layout_a);
    int stride_a0 = is_a_row ? stride_ak : stride_am;
    int stride_a1 = is_a_row ? stride_am : stride_ak;
    int step_a0   = is_a_row ? stride_rep_k : stride_rep_m;
    Value* off_a0 = is_a_row ? offset_a_k_[layout] : offset_a_m_[layout];
    Value* off_a1 = is_a_row ? offset_a_m_[layout] : offset_a_k_[layout];
    Value* phase_a = builder_->CreateURem(builder_->CreateUDiv(off_a1, builder_->getInt32(per_phase_a)),
                                          builder_->getInt32(max_phase_a));
    int num_ptr_a = std::max<int>(2*per_phase_a*max_phase_a / step_a0, 1);
    std::vector<Value*> off_a(num_ptr_a);
    for(int i = 0; i < num_ptr_a; i++){
      Value* off_a0i = builder_->CreateAdd(off_a0, builder_->getInt32(i*(is_a_row?4:stride_rep_m)));
      off_a0i = builder_->CreateExactUDiv(off_a0i, _a_vec);
      off_a0i = builder_->CreateXor(off_a0i, phase_a);
      off_a0i = builder_->CreateMul(off_a0i, _a_vec);
      off_a[i] = builder_->CreateAdd(builder_->CreateMul(off_a0i, builder_->getInt32(stride_a0)),
                                     builder_->CreateMul(off_a1, builder_->getInt32(stride_a1)));
    }

    int per_phase_b = swizzle_->get_per_phase(layout_b);
    int max_phase_b = swizzle_->get_max_phase(layout_b);
    int stride_b0 = is_b_row ? stride_bn : stride_bk;
    int stride_b1 = is_b_row ? stride_bk : stride_bn;
    int step_b0   = is_b_row ? stride_rep_n : stride_rep_k;
    Value* off_b0 = is_b_row ? offset_b_n_[layout] : offset_b_k_[layout];
    Value* off_b1 = is_b_row ? offset_b_k_[layout] : offset_b_n_[layout];
    Value* phase_b = builder_->CreateURem(builder_->CreateUDiv(off_b1, builder_->getInt32(per_phase_b)),
                                          builder_->getInt32(max_phase_b));
    int num_ptr_b = std::max<int>(2*per_phase_b*max_phase_b / step_b0, 1);
    std::vector<Value*> off_b(num_ptr_b);
    for(int i = 0; i < num_ptr_b; i++){
      Value* off_b0i = builder_->CreateAdd(off_b0, builder_->getInt32(i*(is_b_row?stride_rep_n:4)));
      off_b0i = builder_->CreateExactUDiv(off_b0i, _b_vec);
      off_b0i = builder_->CreateXor(off_b0i, phase_b);
      off_b0i = builder_->CreateMul(off_b0i, _b_vec);
      off_b[i] = builder_->CreateAdd(builder_->CreateMul(off_b0i, builder_->getInt32(stride_b0)),
                                     builder_->CreateMul(off_b1, builder_->getInt32(stride_b1)));
    }
    builder_->SetInsertPoint(CurrBB);

    std::vector<Value*> ptr_a(num_ptr_a);
    std::vector<Value*> ptr_b(num_ptr_b);
    std::map<std::pair<int, int>, std::pair<Value*, Value*>> has, hbs;
    for(int i = 0; i < num_ptr_a; i++)
      ptr_a[i] = builder_->CreateGEP(shmems_[A], off_a[i]);
    for(int i = 0; i < num_ptr_b; i++)
      ptr_b[i] = builder_->CreateGEP(shmems_[B], off_b[i]);
    for(auto& x: fcs){
      std::vector<Value *>& fc = x.second;
      for(unsigned m = 0; m < layout->rep(0)/2*shapes[0]/layout->spt(0); m++)
      for(unsigned n = 0; n < layout->rep(1)/2*shapes[1]/layout->spt(1); n++){
      for(unsigned K = 0; K < NK; K += 4){
        if(has.find({m, K}) == has.end()){
          Value* ptra = ptr_a[(is_a_row ? K/4 : m) % num_ptr_a];
          int stepam = is_a_row ? m : m / (num_ptr_a)*(num_ptr_a);
          int stepak = is_a_row ? K / (num_ptr_a*a_vec)*(num_ptr_a*a_vec) : K;
          Value* pa =  builder_->CreateGEP(ptra, builder_->getInt32(stepam*stride_rep_m*stride_am + stepak*stride_ak));
          Value* ha = builder_->CreateLoad(builder_->CreateBitCast(pa, PointerType::get(VectorType::get(builder_->getInt32Ty(), a_vec/2), 3)));
          Value *ha00 = builder_->CreateBitCast(builder_->CreateExtractElement(ha, builder_->getInt32(0)), fp16x2_ty);
          Value *ha01 = builder_->CreateBitCast(builder_->CreateExtractElement(ha, builder_->getInt32(1)), fp16x2_ty);
          has[{m, K}]   = {ha00, ha01};
          if(a_vec > 4){
            Value *ha10 = builder_->CreateBitCast(builder_->CreateExtractElement(ha, builder_->getInt32(2)), fp16x2_ty);
            Value *ha11 = builder_->CreateBitCast(builder_->CreateExtractElement(ha, builder_->getInt32(3)), fp16x2_ty);
            if(is_a_row)
              has[{m, K+4}] = {ha10, ha11};
            else
              has[{m+1, K}] = {ha10, ha11};
          }
        }
        if(hbs.find({n, K}) == hbs.end()){
          Value* ptrb = ptr_b[(is_b_row? n : K/4) % num_ptr_b];
          int stepbn = is_b_row ? n / (num_ptr_b)*(num_ptr_b) : n;
          int stepbk = is_b_row ? K : K / (num_ptr_b*b_vec)*(num_ptr_b*b_vec);

          Value* pb =  builder_->CreateGEP(ptrb, builder_->getInt32(stepbn*stride_rep_n*stride_bn + stepbk*stride_bk));
          Value* hb = builder_->CreateLoad(builder_->CreateBitCast(pb, PointerType::get(VectorType::get(builder_->getInt32Ty(), b_vec/2), 3)));
          Value *hb00 = builder_->CreateBitCast(builder_->CreateExtractElement(hb, builder_->getInt32(0)), fp16x2_ty);
          Value *hb01 = builder_->CreateBitCast(builder_->CreateExtractElement(hb, builder_->getInt32(1)), fp16x2_ty);
          hbs[{n, K}]   = {hb00, hb01};
          if(b_vec > 4){
            Value *hb10 = builder_->CreateBitCast(builder_->CreateExtractElement(hb, builder_->getInt32(2)), fp16x2_ty);
            Value *hb11 = builder_->CreateBitCast(builder_->CreateExtractElement(hb, builder_->getInt32(3)), fp16x2_ty);
            if(is_b_row)
              hbs[{n+1, K}] = {hb10, hb11};
            else
              hbs[{n, K+4}] = {hb10, hb11};
          }
        }
        std::vector<size_t> idx = {
            (m*2 + 0) + (n*4 + 0)*ld_fc,
            (m*2 + 0) + (n*4 + 1)*ld_fc,
            (m*2 + 1) + (n*4 + 0)*ld_fc,
            (m*2 + 1) + (n*4 + 1)*ld_fc,
            (m*2 + 0) + (n*4 + 2)*ld_fc,
            (m*2 + 0) + (n*4 + 3)*ld_fc,
            (m*2 + 1) + (n*4 + 2)*ld_fc,
            (m*2 + 1) + (n*4 + 3)*ld_fc
          };
          auto ha = has[{m, K}];
          auto hb = hbs[{n, K}];
          Value *nc = builder_->CreateCall(mma_fn,  std::vector<llvm::Value*>{ha.first, ha.second, hb.first, hb.second, fc[idx[0]], fc[idx[1]], fc[idx[2]], fc[idx[3]], fc[idx[4]], fc[idx[5]], fc[idx[6]], fc[idx[7]]});
          fc[idx[0]] = builder_->CreateExtractValue(nc, std::vector<unsigned>{0});
          fc[idx[1]] = builder_->CreateExtractValue(nc, std::vector<unsigned>{1});
          fc[idx[2]] = builder_->CreateExtractValue(nc, std::vector<unsigned>{2});
          fc[idx[3]] = builder_->CreateExtractValue(nc, std::vector<unsigned>{3});
          fc[idx[4]] = builder_->CreateExtractValue(nc, std::vector<unsigned>{4});
          fc[idx[5]] = builder_->CreateExtractValue(nc, std::vector<unsigned>{5});
          fc[idx[6]] = builder_->CreateExtractValue(nc, std::vector<unsigned>{6});
          fc[idx[7]] = builder_->CreateExtractValue(nc, std::vector<unsigned>{7});
        }
      }
    }

    // write back
    unsigned i = 0;
    for(indices_t idx: idxs_.at(dot)){
      std::vector<Value*> key(idx.size() - 2);
      std::copy(idx.begin() + 2, idx.end(), key.begin());
      if(i >= fcs.at(key).size())
        i = 0;
      vals_[dot][idx] = fcs.at(key)[i++];
    };

  }
  else{
    analysis::mma_layout* layout = layouts_->get(dot)->to_mma884();
    analysis::shared_layout* layout_a = (analysis::shared_layout*)layouts_->get(dot->get_operand(0));
    analysis::shared_layout* layout_b = (analysis::shared_layout*)layouts_->get(dot->get_operand(1));


    bool is_a_row = ord_a[0] == 1;
    bool is_b_row = ord_b[0] == 1;
    std::string a_trans = is_a_row ? "" : ".trans";
    std::string b_trans = is_b_row ? ".trans" : "";
    int stride_a_m = is_a_row ? shape_a[1] : 1;
    int stride_a_k = is_a_row ? 1 : shape_a[0];
    int stride_b_n = is_b_row ? 1 : shape_b[0];
    int stride_b_k = is_b_row ? shape_b[1] : 1;
    int lda = is_a_row ? stride_a_m : stride_a_k;
    int ldb = is_b_row ? stride_b_k : stride_b_n;

    Type *fp32_ty = builder_->getFloatTy();
    Type *fp16x2_ty = VectorType::get(builder_->getHalfTy(), 2);
    Type *fp16x2_pack4_ty = StructType::get(*ctx_, std::vector<llvm::Type*>{fp16x2_ty, fp16x2_ty, fp16x2_ty, fp16x2_ty});
    Type *fp32_pack4_ty = StructType::get(*ctx_, std::vector<llvm::Type*>{fp32_ty, fp32_ty, fp32_ty, fp32_ty});
    FunctionType *ld_x4_ty = FunctionType::get(fp16x2_pack4_ty, std::vector<llvm::Type*>{PointerType::get(builder_->getHalfTy(), 3)}, false);

    // left-hand-side values
    std::map<std::pair<int, int>, Value*> pTAs;
    std::map<std::pair<unsigned, unsigned>, std::pair<Value*, Value*>> ha;
    std::map<std::pair<unsigned, unsigned>, Value*> hb;


    BasicBlock* CurrBB = builder_->GetInsertBlock();
    BasicBlock* FirstBB = &CurrBB->getParent()->getEntryBlock();
    builder_->SetInsertPoint(FirstBB->getTerminator());
    Value* warp_size = builder_->getInt32(32);
    Value* thread = tgt_->get_local_id(mod_, *builder_, 0);
    Value *lane = builder_->CreateURem(thread, warp_size);
    Value *warp = builder_->CreateUDiv(thread, warp_size);
    Value *warp_id_0 = builder_->CreateURem(warp, builder_->getInt32(layout->wpt(0)));
    Value *warp_id_12 = builder_->CreateUDiv(warp, builder_->getInt32(layout->wpt(0)));
    Value *warp_id_1 = builder_->CreateURem(warp_id_12, builder_->getInt32(layout->wpt(1)));
    std::vector<Value *>& fc = fcs.begin()->second;

    int per_phase_a = swizzle_->get_per_phase(layout_a);
    int max_phase_a = swizzle_->get_max_phase(layout_a);
    int num_ptr_row_a = !is_a_row ? std::min<int>(shapes[0] / layout->spt(0), max_phase_a): 1;
    int num_ptr_col_a = 2;
    Value *a_base = builder_->CreateURem(lane, builder_->getInt32(8));
    Value *a_phase = builder_->CreateURem(builder_->CreateUDiv(a_base, builder_->getInt32(per_phase_a)), builder_->getInt32(max_phase_a));
    Value *a_row0 = builder_->CreateAdd(builder_->CreateURem(builder_->CreateUDiv(lane, builder_->getInt32(8)), builder_->getInt32(2)),
                                        builder_->CreateMul(warp_id_0, builder_->getInt32(2)));
    Value *a_col0 = builder_->CreateUDiv(lane, builder_->getInt32(16));
    Value* a_off = builder_->CreateMul(a_base, builder_->getInt32(lda));
    std::map<std::pair<int,int>, Value*> a_offs;
    for(size_t r = 0; r < num_ptr_row_a; r++){
      Value *off_a_m = builder_->CreateAdd(a_row0, builder_->getInt32(2*layout->wpt(0)*r));
      off_a_m = is_a_row ? off_a_m : builder_->CreateXor(off_a_m, a_phase);
      for(size_t c = 0; c < num_ptr_col_a; c++){
        Value *off_a_k = builder_->CreateAdd(a_col0, builder_->getInt32(2*c));
        off_a_k = is_a_row ? builder_->CreateXor(off_a_k, a_phase) : off_a_k;
        a_offs[{r, c}] = builder_->CreateAdd(a_off,
                         builder_->CreateAdd(builder_->CreateMul(off_a_k, builder_->getInt32(8*stride_a_k)),
                                             builder_->CreateMul(off_a_m, builder_->getInt32(8*stride_a_m))));
      }
    }

    std::map<std::pair<int,int>, Value*> pTBs;
    int per_phase_b = swizzle_->get_per_phase(layout_b);
    int max_phase_b = swizzle_->get_max_phase(layout_b);
    int num_ptr_row_b = 2;
    int num_ptr_col_b = is_b_row ? std::min<int>(shapes[1] / layout->spt(1), max_phase_b) : 1;
    Value *b_base = builder_->CreateURem(lane, builder_->getInt32(8));
    Value *b_phase = builder_->CreateURem(builder_->CreateUDiv(b_base, builder_->getInt32(per_phase_b)), builder_->getInt32(max_phase_b));
    Value *b_row0 = builder_->CreateURem(builder_->CreateUDiv(lane, builder_->getInt32(8)), builder_->getInt32(2));
    Value *b_col0 = builder_->CreateAdd(builder_->CreateMul(builder_->CreateUDiv(lane, builder_->getInt32(16)), builder_->getInt32(layout->wpt(1))),
                                        builder_->CreateMul(warp_id_1, builder_->getInt32(1)));
    Value *off_b = builder_->CreateMul(b_base, builder_->getInt32(ldb));
    std::map<std::pair<int,int>, Value*> b_offs;
    for(size_t r = 0; r < num_ptr_row_b; r++){
      Value *off_b_k = builder_->CreateAdd(b_row0, builder_->getInt32(r*2));
      off_b_k = is_b_row ? off_b_k : builder_->CreateXor(off_b_k, b_phase);
      for(size_t c = 0; c < num_ptr_col_b; c++){
        Value *off_b_n = builder_->CreateAdd(b_col0, builder_->getInt32(c*2*layout->wpt(1)));
        off_b_n = is_b_row ? builder_->CreateXor(off_b_n, b_phase) : off_b_n;
        b_offs[{r, c}] = builder_->CreateAdd(off_b,
                         builder_->CreateAdd(builder_->CreateMul(off_b_n, builder_->getInt32(8*stride_b_n)),
                                             builder_->CreateMul(off_b_k, builder_->getInt32(8*stride_b_k))));
      }
    }

    builder_->SetInsertPoint(CurrBB);
    Value *pTA = shmems_[A];
    for(size_t r = 0; r < num_ptr_row_a; r++)
    for(size_t c = 0; c < num_ptr_col_a; c++)
      pTAs[{r, c}] = builder_->CreateGEP(pTA, {a_offs[{r,c}]});
    Value *pTB = shmems_[B];
    for(size_t r = 0; r < num_ptr_row_b; r++)
    for(size_t c = 0; c < num_ptr_col_b; c++)
      pTBs[{r, c}] = builder_->CreateGEP(pTB, {b_offs[{r,c}]});

    FunctionType *mma_ty = FunctionType::get(fp32_pack4_ty, std::vector<llvm::Type*>{fp16x2_ty, fp16x2_ty, fp16x2_ty, fp16x2_ty, fp16x2_ty, fp16x2_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty}, false);
    InlineAsm *mma_fn = InlineAsm::get(mma_ty, "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                                               "{$0, $1, $2, $3}, "
                                               "{$4, $5, $6, $7}, "
                                               "{$8, $9}, "
                                               "{$10, $11, $12, $13};", "=f,=f,=f,=f,r,r,r,r,r,r,0,1,2,3", false);
    unsigned num_rep_0 = shapes[0] / layout->spt(0);
    unsigned num_rep_1 = shapes[1] / layout->spt(1);
    for(unsigned K = 0; K < NK; K += 16)
    for(unsigned pack_i = 0; pack_i < num_rep_0; pack_i++)
    for(unsigned pack_j = 0; pack_j < num_rep_1; pack_j++){
      if(ha.find({pack_i, K}) == ha.end()){
        InlineAsm *ld_a0_fn = InlineAsm::get(ld_x4_ty, "ldmatrix.sync.aligned.m8n8.x4" + a_trans + ".shared.b16 "
                                                  "{$0, $1, $2, $3}, [$4 + " + std::to_string(pack_i/num_ptr_row_a*num_ptr_row_a*layout->wpt(0)*layout->spw(0)*2*stride_a_m) + "];", "=r,=r,=r,=r,r", false);
        Value *haa = builder_->CreateCall(ld_x4_ty, ld_a0_fn, {pTAs[{pack_i % num_ptr_row_a, K/16 % num_ptr_col_a}]});
        Value *ha0 = builder_->CreateExtractValue(haa, std::vector<unsigned>{0});
        Value *ha1 = builder_->CreateExtractValue(haa, std::vector<unsigned>{1});
        Value *ha2 = builder_->CreateExtractValue(haa, std::vector<unsigned>{2});
        Value *ha3 = builder_->CreateExtractValue(haa, std::vector<unsigned>{3});
        ha[{pack_i, K}] = std::make_pair(ha0, ha1);
        ha[{pack_i, K+8}] = std::make_pair(ha2, ha3);
      }
      if(hb.find({pack_j, K})==hb.end()){
        InlineAsm *ld_b_fn = InlineAsm::get(ld_x4_ty, "ldmatrix.sync.aligned.m8n8.x4" + b_trans + ".shared.b16 "
                                                  "{$0, $1, $2, $3}, [$4 + " + std::to_string(pack_j/(2*num_ptr_col_b)*(2*num_ptr_col_b)*layout->wpt(1)*layout->spw(1)*2*stride_b_n
                                                                                              + K/(16*num_ptr_row_b)*(16*num_ptr_row_b)*stride_b_k) + "];", "=r,=r,=r,=r,r", false);
        Value *hbb = builder_->CreateCall(ld_x4_ty, ld_b_fn, {pTBs[{(K/16 % num_ptr_row_b), (pack_j%(2*num_ptr_col_b))/2}]});
        Value *hb0 = builder_->CreateExtractValue(hbb, std::vector<unsigned>{0});
        Value *hb1 = builder_->CreateExtractValue(hbb, std::vector<unsigned>{1});
        Value *hb2 = builder_->CreateExtractValue(hbb, std::vector<unsigned>{2});
        Value *hb3 = builder_->CreateExtractValue(hbb, std::vector<unsigned>{3});
        hb[{pack_j, K}] = hb0;
        hb[{pack_j+1, K}] = hb2;
        hb[{pack_j, K+8}] = hb1;
        hb[{pack_j+1, K+8}] = hb3;
      }
      unsigned cols_per_thread = num_rep_0 * 2;
      std::vector<size_t> idx = {
        (pack_i*2 + 0) + (pack_j*2 + 0)*cols_per_thread,
        (pack_i*2 + 0) + (pack_j*2 + 1)*cols_per_thread,
        (pack_i*2 + 1) + (pack_j*2 + 0)*cols_per_thread,
        (pack_i*2 + 1) + (pack_j*2 + 1)*cols_per_thread
      };
      Value *nc = builder_->CreateCall(mma_ty, mma_fn, {ha[{pack_i, K}].first, ha[{pack_i, K}].second,ha[{pack_i, K+8}].first, ha[{pack_i, K+8}].second,
                                                        hb[{pack_j, K}], hb[{pack_j, K+8}],
                                                        fc[idx[0]], fc[idx[1]], fc[idx[2]], fc[idx[3]]});
      fc[idx[0]] = builder_->CreateExtractValue(nc, std::vector<unsigned>{0});
      fc[idx[1]] = builder_->CreateExtractValue(nc, std::vector<unsigned>{1});
      fc[idx[2]] = builder_->CreateExtractValue(nc, std::vector<unsigned>{2});
      fc[idx[3]] = builder_->CreateExtractValue(nc, std::vector<unsigned>{3});
    }

    // write back
    unsigned i = 0;
    for(indices_t idx: idxs_.at(dot)){
      std::vector<Value*> key(idx.size() - 2);
      std::copy(idx.begin() + 2, idx.end(), key.begin());
      if(i >= fcs.at(key).size())
        i = 0;
      vals_[dot][idx] = fcs.at(key)[i++];
    };
  }
}
void generator::visit_scanline_dot(ir::dot_inst* dot, ir::value* A, ir::value* B, ir::value* D, unsigned NK,
                                   Type *c_ty, Function *f_mul_add) {
  throw std::runtime_error("TODO: v1.0 not complete");
//  TA->set_vector_size(TD->axis(0).contiguous);
//  TB->set_vector_size(TD->axis(1).contiguous);
//  for_each(dot, [&](indices_t idx, int){
//    Value *res = TD->get_value(idx);
//    for(unsigned K = 0; K < NK; ++K){
//      // input indices
//      indices_t a_idx = {idx[0], builder_->getInt32(K)};
//      indices_t b_idx = {builder_->getInt32(K), idx[1]};
//      // add batching dimension
//      for(size_t i = 2; i < idx.size(); i++){
//        a_idx.insert(a_idx.end(), idx[i]);
//        b_idx.insert(b_idx.end(), idx[i]);
//      }
//      // load value
//      Value *a = TA->get_value(a_idx);
//      Value *b = TB->get_value(b_idx);
//      if(a->getType() != c_ty)
//        a = builder_->CreateFPCast(a, c_ty);
//      if(b->getType() != c_ty)
//        b = builder_->CreateFPCast(b, c_ty);
//      res = builder_->CreateCall(f_mul_add, std::vector<llvm::Value*>{a, b, res});
//    }
//    set_value(dot, idx, res);
//  });
}

void generator::visit_outer_dot(ir::dot_inst* dot, ir::value *A, ir::value *B, ir::value *D, unsigned NK,
                                Type *c_ty, Function *f_mul_add) {
  throw std::runtime_error("TODO: v1.0 not complete");
//  for_each(dot, [&](indices_t idx, int){
//    Value *res = TD->get_value(idx);
//    indices_t a_idx = {idx[0], builder_->getInt32(0)};
//    indices_t b_idx = {builder_->getInt32(0), idx[1]};
//    std::swap(a_idx[0], a_idx[1]);
//    std::swap(b_idx[0], b_idx[1]);
//    Value *a = TA->get_value(a_idx);
//    Value *b = TB->get_value(b_idx);
//    if(a->getType() != c_ty)
//      a = builder_->CreateFPCast(a, c_ty);
//    if(b->getType() != c_ty)
//      b = builder_->CreateFPCast(b, c_ty);
//    res = builder_->CreateCall(f_mul_add, std::vector<llvm::Value*>{a, b, res});
//    set_value(dot, idx, res);
//  });
}

void generator::visit_dot_inst(ir::dot_inst* dot) {
  Function *fn = builder_->GetInsertBlock()->getParent();

  Module *module = fn->getParent();
  ir::value *A = dot->get_operand(0);
  ir::value *B = dot->get_operand(1);
  ir::value *D = dot->get_operand(2);

  Type *c_ty = llvm_type(D->get_type()->get_scalar_ty(), *ctx_);
  Function *f_mul_add = Intrinsic::getDeclaration(module, Intrinsic::fmuladd, std::vector<llvm::Type*>{c_ty});
  auto A_shapes = A->get_type()->get_tile_shapes();
  size_t red_axis = 1;
  unsigned NK = A_shapes[red_axis];

  if(NK != 1) {
    if(layouts_->get(dot)->to_mma884())
      visit_hmma_dot(dot, A, B, D, NK);
    else
      visit_scanline_dot(dot, A, B, D, NK, c_ty, f_mul_add);
  }
  else {
    visit_outer_dot(dot, A, B, D, NK, c_ty, f_mul_add);
  }
}

void generator::visit_trans_inst(ir::trans_inst* trans) {
  throw std::runtime_error("not supported");
}

void generator::visit_sqrt_inst(ir::sqrt_inst* x) {
  for(indices_t idx: idxs_.at(x)){
    Value *val = vals_[x->get_operand(0)][idx];
    Value *ret = builder_->CreateIntrinsic(Intrinsic::sqrt,
                                           std::vector<llvm::Type*>{val->getType()},
                                           std::vector<llvm::Value*>{val});
    vals_[x][idx] = ret;
  }
}

inline Value* shared_offset(llvm::IRBuilder<> &builder, const std::vector<unsigned>& shapes, const std::vector<int>& order, indices_t idx){
  // strides
  std::vector<Value*> strides(shapes.size(), builder.getInt32(0));
  strides[order[0]] = builder.getInt32(1);
  for(size_t i = 1; i < idx.size(); i++)
    strides[order[i]] = builder.CreateMul(strides[order[i-1]], builder.getInt32(shapes[order[i-1]]));
  // result
  Value *result = builder.getInt32(0);
  for(size_t i = 0; i < idx.size(); i++)
    result = builder.CreateAdd(result, builder.CreateMul(idx[i], strides[i]));
  return result;
}

void generator::visit_reduce_inst(ir::reduce_inst* x) {
  std::map<indices_t, Value*> partial;
  ir::value *arg = x->get_operand(0);
  Type *ty = llvm_type(x->get_type(), builder_->getContext());
  ir::reduce_inst::op_t op = x->get_op();
  unsigned axis = x->get_axis();

  Type *fp32_ty = builder_->getFloatTy();
  FunctionType *fmaxmin_ty = FunctionType::get(fp32_ty, std::vector<llvm::Type*>{fp32_ty, fp32_ty}, false);
  InlineAsm *fmin = InlineAsm::get(fmaxmin_ty, "min.ftz.f32 $0, $1, $2;", "=f,f,f", false);
  InlineAsm *fmax = InlineAsm::get(fmaxmin_ty, "max.ftz.f32 $0, $1, $2;", "=f,f,f", false);

  auto accumulate = [&](Value* x, Value *y) -> Value* {
    switch(op) {
      case ir::reduce_inst::ADD: return builder_->CreateAdd(x, y);
      case ir::reduce_inst::SUB: return builder_->CreateSub(x, y);
      case ir::reduce_inst::MAX:{
        if(x->getType()->isIntegerTy())
          return builder_->CreateSelect(builder_->CreateICmpSGE(x, y), x, y);
        else
          return builder_->CreateMaxNum(x, y);
      }
      case ir::reduce_inst::MIN:{
        if(x->getType()->isIntegerTy())
          return builder_->CreateSelect(builder_->CreateICmpSLE(x, y), x, y);
        else
          return builder_->CreateMinNum(x, y);
      }
      case ir::reduce_inst::FADD: return builder_->CreateFAdd(x, y);
      case ir::reduce_inst::FSUB: return builder_->CreateFSub(x, y);
      case ir::reduce_inst::FMAX: return builder_->CreateCall(fmax, std::vector<llvm::Value*>{x, y});
      case ir::reduce_inst::FMIN: return builder_->CreateCall(fmin, std::vector<llvm::Value*>{x, y});
      default: assert(false); return nullptr;
    }
  };

  Value *neutral;
  switch(op) {
    case ir::reduce_inst::ADD: neutral = builder_->getInt32(0); break;
    case ir::reduce_inst::SUB: neutral = builder_->getInt32(0); break;
    case ir::reduce_inst::MAX: neutral = builder_->getInt32(INT32_MIN); break;
    case ir::reduce_inst::MIN: neutral = builder_->getInt32(INT32_MAX); break;
    case ir::reduce_inst::FADD: neutral = ConstantFP::get(ty, 0); break;
    case ir::reduce_inst::FSUB: neutral = ConstantFP::get(ty, 0); break;
    case ir::reduce_inst::FMAX: neutral = ConstantFP::get(ty, -INFINITY); break;
    case ir::reduce_inst::FMIN: neutral = ConstantFP::get(ty, INFINITY); break;
    default: assert(false); break;
  }


  analysis::data_layout* arg_layout = layouts_->get(arg);
  if(auto* L = dynamic_cast<analysis::scanline_layout*>(arg_layout)){
    bool can_optimize = L->get_rank() == 1;

    if(can_optimize){
      Value *thread_acc = nullptr;
      // reduce within thread
      for(indices_t idx: idxs_.at(arg)){
        Value *current = vals_[arg][idx];
        if(thread_acc == nullptr)
          thread_acc = current;
        else
          thread_acc = accumulate(thread_acc, current);
      }
      // reduce within wrap
      FunctionType *fn_ty = FunctionType::get(thread_acc->getType(), std::vector<llvm::Type*>{thread_acc->getType(), builder_->getInt32Ty()}, false);
      InlineAsm *shfl_xor = InlineAsm::get(fn_ty, "shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;", "=f,f,r", false);
      Value *warp_acc = thread_acc;
      for(int i = 16; i > 0; i >>= 1)
        warp_acc = accumulate(warp_acc, builder_->CreateCall(shfl_xor, std::vector<llvm::Value*>{warp_acc, builder_->getInt32(i)}));
      // shared memory pointer
      unsigned addr_space = shmem_->getType()->getPointerAddressSpace();
      Type *res_ty = ty;
      Value *sh_mem_ptr = builder_->CreateBitCast(shmem_, PointerType::get(res_ty, addr_space));
      Value* thread_id = tgt_->get_local_id(builder_->GetInsertBlock()->getModule(), *builder_, 0);
      Value* warp_id = builder_->CreateUDiv(thread_id, builder_->getInt32(32));
      Value* lane_id = builder_->CreateURem(thread_id, builder_->getInt32(32));
      Value *write_ptr = builder_->CreateGEP(sh_mem_ptr, warp_id);
      // store warp result in shared memory
      tgt_->add_barrier(mod_, *builder_);
      builder_->CreateStore(neutral, builder_->CreateGEP(sh_mem_ptr, lane_id));
      tgt_->add_barrier(mod_, *builder_);
      builder_->CreateStore(warp_acc, write_ptr);
      tgt_->add_barrier(mod_, *builder_);
      // accumulate all warps
      Value *load_ptr = builder_->CreateGEP(sh_mem_ptr, thread_id);
      Value* is_first_warp = builder_->CreateICmpEQ(warp_id, builder_->getInt32(0));
      BasicBlock* bb_final_acc = BasicBlock::Create(*ctx_, "bb_final_acc", builder_->GetInsertBlock()->getParent());
      BasicBlock* bb_final_acc_done = BasicBlock::Create(*ctx_, "bb_final_acc_done", builder_->GetInsertBlock()->getParent());
      builder_->CreateCondBr(is_first_warp, bb_final_acc, bb_final_acc_done);
      builder_->SetInsertPoint(bb_final_acc);
      Value* ret = builder_->CreateLoad(load_ptr);
      for(int i = (num_warps_+1)/2; i > 0; i >>= 1){
        Value *current = builder_->CreateCall(shfl_xor, std::vector<llvm::Value*>{ret, builder_->getInt32(i)});
        ret = accumulate(ret, current);
      }
      builder_->CreateStore(ret, load_ptr);
      builder_->CreateBr(bb_final_acc_done);
      // store first warp done
      builder_->SetInsertPoint(bb_final_acc_done);
      // write back
      tgt_->add_barrier(mod_, *builder_);
      ret = builder_->CreateLoad(sh_mem_ptr);
      for(indices_t idx: idxs_.at(x))
        vals_[x][idx] = ret;
      return;
    }
  }

  // reduce within thread
  for(indices_t idx: idxs_.at(arg)){
    indices_t pidx = idx;
    pidx[axis] = builder_->getInt32(0);
    Value *current = vals_[arg][idx];
    // current partial result is not initialized -- create
    if(partial.find(pidx) == partial.end())
      partial[pidx] = current;
    // current partial result is initialized -- accumulate
    else
      partial[pidx] = accumulate(partial[pidx], current);
  };

  // reduce within blocks
  auto shapes = x->get_type()->get_tile_shapes();
  auto ord = layouts_->get(x)->get_order();
  unsigned depth = shapes[axis];

  unsigned addr_space = shmem_->getType()->getPointerAddressSpace();
  Type *res_ty = ty;
  Value *base_ptr = builder_->CreateBitCast(shmem_, PointerType::get(res_ty, addr_space));
  for(auto& x: partial) {
    // current element being computed
    Value *lane = axes_.at(a_axes_->get(arg, axis)).thread_id;
    Value *&result = x.second;
    indices_t write_idx = x.first;
    write_idx[axis] = lane;
    // shared memory write  pointer
    Value *write_offset = shared_offset(*builder_, shapes, ord, write_idx);
    Value *write_ptr = builder_->CreateGEP(base_ptr, write_offset);
    // initialize shared memory
    tgt_->add_barrier(mod_, *builder_);
    builder_->CreateStore(result, write_ptr);
    // build result
    for(unsigned i = depth/2; i > 0; i >>= 1){
      // current indices
      indices_t current(write_idx.size(), builder_->getInt32(0));
      current[axis] = builder_->getInt32(i);
      // shared memory offset
      Value *read_offset = shared_offset(*builder_, shapes, ord, current);
      Value *is_active = builder_->CreateICmpULT(lane, builder_->getInt32(i));
      read_offset = builder_->CreateSelect(is_active, read_offset, builder_->getInt32(0));
      // shared memory read pointer
      Value *read_ptr = builder_->CreateGEP(write_ptr, read_offset);
      tgt_->add_barrier(mod_, *builder_);
      Value *next = builder_->CreateLoad(read_ptr);
      // accumulate
      result = accumulate(result, next);
      // write back
      tgt_->add_barrier(mod_, *builder_);
      builder_->CreateStore(result, write_ptr);
    }
  }
  tgt_->add_barrier(mod_, *builder_);

  // write back
  for(indices_t idx: idxs_.at(arg)){
    indices_t red_idx = idx;
    red_idx.insert(red_idx.begin() + axis, builder_->getInt32(0));
    Value *read_offset = shared_offset(*builder_, shapes, ord,  red_idx);
    Value *read_ptr = builder_->CreateGEP(base_ptr, read_offset);
    vals_[x][idx] = builder_->CreateLoad(read_ptr);
  };
}

void generator::visit_select_inst(ir::select_inst* x) {
  for(indices_t idx: idxs_.at(x)){
    vals_[x][idx] = builder_->CreateSelect(vals_[x->get_operand(0)][idx],
                                           vals_[x->get_operand(1)][idx],
                                           vals_[x->get_operand(2)][idx]);
  }
}

void generator::visit_recoalesce_inst(ir::recoalesce_inst* rc) {
  ir::value *op = rc->get_operand(0);
  ir::tile_type::tile_shapes_t shape = rc->get_type()->get_tile_shapes();
  // pointer to temporary shared memory
  Type *ty = llvm_type(rc->get_type()->get_scalar_ty(), *ctx_);
  // layout
  analysis::mma_layout* in_layout = layouts_->get(op)->to_mma884();
  analysis::scanline_layout* out_layout = layouts_->get(rc)->to_scanline();
  // Orders
  auto ord = layouts_->get(rc)->to_scanline()->get_order();
  Value *base;
  base = builder_->CreateGEP(shmem_, builder_->getInt32(alloc_->offset(layouts_->get(layouts_->tmp(rc)))));
  base = builder_->CreateBitCast(base, PointerType::get(ty, 3));
  Value *ld = builder_->getInt32(shape[ord[0]]);
  auto in_ord0 = axes_.at(a_axes_->get(op, ord[0])).values;
  auto in_ord1 = axes_.at(a_axes_->get(op, ord[1])).values;
  auto out_ord0 = axes_.at(a_axes_->get(rc, ord[0])).values;
  auto out_ord1 = axes_.at(a_axes_->get(rc, ord[1])).values;
  indices_t idx(2);
  int in_outer = in_layout->spt(ord[1]);
  int in_rep   = in_layout->rep(ord[1]);
  int out_outer = out_layout->mts(ord[1]) * out_layout->nts(ord[1]);
  size_t max_outer = std::max(in_outer, out_outer);
  size_t out_ratio = std::max<size_t>(out_outer/in_outer, 1);
  size_t in_ratio = std::max<size_t>(in_outer/out_outer, 1);
  for(size_t j = 0; j < shape[ord[1]]/max_outer; j++){
    tgt_->add_barrier(mod_, *builder_);
    for(size_t k = 0; k < in_rep*out_ratio; k++)
    for(size_t i = 0; i < in_ord0.size(); i++){
      idx[ord[0]] = in_ord0[i];
      idx[ord[1]] = in_ord1[j*in_rep*out_ratio + k];
      Value *off = builder_->CreateAdd(idx[ord[0]], builder_->CreateMul(in_ord1[k], ld));
      Value *ptr = builder_->CreateGEP(base, off);
      builder_->CreateStore(vals_[op][idx], ptr);
    }
    tgt_->add_barrier(mod_, *builder_);
    for(size_t k = 0; k < in_ratio; k++)
    for(size_t i = 0; i < out_ord0.size(); i++){
      idx[ord[0]] = out_ord0[i];
      idx[ord[1]] = out_ord1[j*in_ratio + k];
      Value *off = builder_->CreateAdd(out_ord0[i], builder_->CreateMul(out_ord1[k], ld));
      Value *ptr  = builder_->CreateGEP(base, off);
      vals_[rc][idx] = builder_->CreateLoad(ptr);
    }
  }
}

void generator::visit_masked_load_async_inst(ir::masked_load_async_inst* x){
  unsigned vector = 1;
  ir::value *ptrs = x->get_pointer_operand();
  ir::value *msks = x->get_mask_operand();
  analysis::shared_layout* out_layout = layouts_->get(x)->to_shared();
  analysis::scanline_layout* in_layout = layouts_->get(ptrs)->to_scanline();
  auto out_order = out_layout->get_order();
  auto in_order = in_layout->get_order();
  // tiles
  if(out_order == in_order)
    vector = in_layout->nts(in_order[0]);
  //
  int dtsize = x->get_type()->get_scalar_ty()->get_primitive_size_in_bits() / 8;
  int num_per_phase = std::max<int>(128 / (in_layout->mts(in_order[0])*vector*dtsize), 1);
  Value *max_phase = builder_->getInt32(8 / num_per_phase);
  //
  auto shapes = x->get_type()->get_tile_shapes();
  //
  int per_thread_ld = in_layout->get_shape()[in_order[0]] / in_layout->mts(in_order[0]);
  int n_shared = std::max<int>(8 / in_layout->mts(in_order[1]), 1);
  std::vector<Value*> shared;
  for(size_t i = 0; i < n_shared; i++){
    indices_t idx = idxs_.at(ptrs).at(i*per_thread_ld);
    // phase
    Value* phase = builder_->CreateUDiv(idx[in_order[1]], builder_->getInt32(num_per_phase));
    phase = builder_->CreateURem(phase, max_phase);
    // off
    Value* off_0  = idx[in_order[0]];
    off_0 = builder_->CreateUDiv(off_0, builder_->getInt32(vector));
    off_0 = builder_->CreateXor(off_0, phase);
    off_0 = builder_->CreateMul(off_0 , builder_->getInt32(vector));
    Value* off_1 = builder_->CreateMul(idx[in_order[1]], builder_->getInt32(shapes[in_order[0]]));
    Value* off = builder_->CreateAdd(off_0, off_1);
    //
    shared.push_back(builder_->CreateGEP(shmems_[x], {off}));
  }
  //
  for(size_t i = 0; i < idxs_.at(ptrs).size(); i += vector){
    auto idx = idxs_[ptrs][i];
    // input ptr info
    GetElementPtrInst *in_gep = dyn_cast<GetElementPtrInst>(vals_[ptrs][idx]);
    Value *in_base = in_gep->getPointerOperand();
    size_t in_off = dyn_cast<ConstantInt>(in_gep->idx_begin())->getValue().getSExtValue()*2*vector;
    Value* out_base = shared[(i / per_thread_ld) % n_shared];
    int out_off_0 = (i / per_thread_ld) / n_shared * n_shared * in_layout->mts(in_order[1]);
    int out_off_1 = i % per_thread_ld;
    int out_off = (out_off_0*shapes[in_order[0]] + out_off_1)*2;
    // asm
    FunctionType *ty = FunctionType::get(builder_->getVoidTy(), {out_base->getType(), in_base->getType()}, false);
    std::string mod = (vector*2 == 16) ? ".cg" : ".ca";
    std::string asm_str = "@$0 cp.async" + mod + ".shared.global [$1 + " + std::to_string(out_off) + "], [$2 + " + std::to_string(in_off) + "], " + std::to_string(vector*2) + ";";
    InlineAsm *iasm = InlineAsm::get(ty, asm_str, "b,r,l", true);
    builder_->CreateCall(iasm, {vals_[msks][idx], out_base, in_base});
  }
}

void generator::visit_copy_to_shared_inst(ir::copy_to_shared_inst* cts) {
  unsigned in_vec = 1;
  ir::value *arg = cts->get_operand(0);
  analysis::shared_layout* out_layout = layouts_->get(cts)->to_shared();
  analysis::scanline_layout* in_layout = layouts_->get(arg)->to_scanline();
  auto out_order = out_layout->get_order();
  auto in_order = in_layout->get_order();
  // tiles
  if(out_order == in_order)
    in_vec = in_layout->nts(in_order[0]);
  int out_vec = swizzle_->get_vec(out_layout);
  int min_vec = std::min<int>(out_vec, in_vec);
  int s = std::max<int>(out_vec / in_vec, 1);
  //
  int per_phase = swizzle_->get_per_phase(out_layout);
  int max_phase = swizzle_->get_max_phase(out_layout);
  //
  int in_ld = in_layout->get_shape()[in_order[0]] / in_layout->mts(in_order[0]);
  int n_shared_1 = std::max<int>(per_phase*max_phase / in_layout->mts(in_order[1]), 1);
  int n_shared_0 = std::max<int>(in_vec    / out_vec, 1);

  BasicBlock* CurrBB = builder_->GetInsertBlock();
  BasicBlock* FirstBB = &CurrBB->getParent()->getEntryBlock();
  auto shapes = cts->get_type()->get_tile_shapes();

  // default implementation
  Value *current = nullptr;
  std::map<std::pair<int, int>, Value*> ptrs;
  for(int i = 0; i < idxs_.at(arg).size(); i++){
    auto idx = idxs_[arg][i];
    Value *in_value = vals_[arg][idx];
    if(i % min_vec == 0)
      current = UndefValue::get(VectorType::get(in_value->getType(), min_vec));
    current = builder_->CreateInsertElement(current, in_value, i % min_vec);
    if(i % min_vec == min_vec - 1){
      unsigned id = i / min_vec;
      // input ptr info
      int id_0 = id % (in_ld/min_vec);
      int id_1 = id / (in_ld/min_vec);
      int off_0 = id_0 / n_shared_0 * n_shared_0 * in_layout->mts(in_order[0]);
      int off_1 = id_1 / n_shared_1 * n_shared_1 * in_layout->mts(in_order[1]);
      int off = (off_1*shapes[in_order[0]] + off_0);
      std::pair<int, int> key = {id_1  % n_shared_1, id_0 % n_shared_0};
      if(ptrs.find(key) == ptrs.end()){
        builder_->SetInsertPoint(FirstBB->getTerminator());
        indices_t idx = idxs_.at(arg).at(key.first*in_ld);
        Value* phase = builder_->CreateUDiv(idx[in_order[1]], builder_->getInt32(per_phase));
        phase = builder_->CreateURem(phase, builder_->getInt32(max_phase));
        Value* off_1 = builder_->CreateMul(idx[in_order[1]], builder_->getInt32(shapes[in_order[0]]));
        Value* off_0  = builder_->CreateAdd(idx[in_order[0]], builder_->getInt32(key.second*out_vec));
        off_0 = builder_->CreateExactUDiv(off_0, builder_->getInt32(min_vec));
        off_0 = builder_->CreateAdd(builder_->CreateMul(builder_->CreateXor(builder_->CreateUDiv(off_0, builder_->getInt32(s)),
                                                                            phase),
                                                        builder_->getInt32(s)),
                                    builder_->CreateURem(off_0, builder_->getInt32(s)));
        off_0 = builder_->CreateMul(off_0 , builder_->getInt32(min_vec));
        Value* off = builder_->CreateAdd(off_0, off_1);
        builder_->SetInsertPoint(CurrBB);
        ptrs[key] = builder_->CreateGEP(shmems_.at(cts), {off});
      }
      Value* ptr = builder_->CreateGEP(ptrs[key], {builder_->getInt32(off)});
      ptr = builder_->CreateBitCast(ptr, current->getType()->getPointerTo(3));
      // asm
      builder_->CreateStore(current, ptr);
    }
  };
}

void generator::visit_copy_from_shared_inst(ir::copy_from_shared_inst* x) {
//  throw std::runtime_error("TODO");
//  for_each(x, [&](indices_t idx, int){
//    set_value(x, idx, get_value(x->get_operand(0), idx));
//  });
}

void generator::visit_barrier_inst(ir::barrier_inst*) {
  Module *module = builder_->GetInsertBlock()->getModule();
  tgt_->add_barrier(module, *builder_);
}

void generator::visit_async_wait_inst(ir::async_wait_inst*) {
  Module *module = builder_->GetInsertBlock()->getModule();
  std::string asm_str = "cp.async.wait_all;";
  InlineAsm *iasm = InlineAsm::get(FunctionType::get(builder_->getVoidTy(), {}), asm_str, "", true);
  builder_->CreateCall(iasm);
  tgt_->add_barrier(module, *builder_);
}

void generator::visit_make_range_dyn(ir::make_range_dyn* x) {
  for(indices_t idx: idxs_.at(x)){
    assert(idx.size() == 1);
    if(idx[0] == builder_->getInt32(0))
      vals_[x][idx] = idx[0];
    else{
      BinaryOperator *bin_add = dyn_cast<BinaryOperator>(idx[0]);
      assert(bin_add);
      vals_[x][idx] = bin_add->getOperand(0);
    }
  }
}

void generator::visit_make_range_sta(ir::make_range_sta* x) {
  for(indices_t idx: idxs_.at(x)){
    assert(idx.size() == 1);
    if(idx[0] == builder_->getInt32(0)){
      vals_[x][idx] = idx[0];
    }
    else{
      BinaryOperator *bin_add = dyn_cast<BinaryOperator>(idx[0]);
      assert(bin_add);
      Value *cst = bin_add->getOperand(1);
      assert(isa<Constant>(cst));
      vals_[x][idx] = cst;
    }
  };
}

void generator::visit_make_range(ir::make_range* x) {
  for(indices_t idx: idxs_.at(x)){
    vals_[x][idx] = idx[0];
  }
}



void generator::visit_undef_value(ir::undef_value *ud) {
  vals_[ud][{}] = llvm::UndefValue::get(llvm_type(ud->get_type(), *ctx_));
}

void generator::visit_constant_int(ir::constant_int *cst){
  Type *ty = llvm_type(cst->get_type()->get_scalar_ty(), *ctx_);
  vals_[cst][{}] = ConstantInt::get(ty, cst->get_value());
}

void generator::visit_constant_fp(ir::constant_fp *cst){
  Type *ty = llvm_type(cst->get_type()->get_scalar_ty(), *ctx_);
  vals_[cst][{}] = ConstantFP::get(ty, cst->get_value());
}

void generator::visit_alloc_const(ir::alloc_const *alloc) {
  unsigned size = ((ir::constant_int*)alloc->get_operand(0))->get_value();
  Type *element_ty = llvm_type(alloc->get_type()->get_pointer_element_ty(), *ctx_);
  Type *array_ty = llvm::ArrayType::get(element_ty, size);
  Value *array = new llvm::GlobalVariable(*mod_, array_ty, false, llvm::GlobalVariable::ExternalLinkage,
                                            nullptr, alloc->get_name(), nullptr, llvm::GlobalVariable::NotThreadLocal, 4);
  vals_[alloc][{}] = builder_->CreateBitCast(array, element_ty->getPointerTo(4));
}


void generator::visit_function(ir::function* fn) {
  LLVMContext &ctx = builder_->getContext();
  FunctionType *fn_ty = (FunctionType*)llvm_type(fn->get_fn_type(), *ctx_);
  if(!tgt_->is_gpu()){
    Type *fn_ret_ty = fn_ty->getReturnType();
    std::vector<Type*> fn_args_ty;
    for(unsigned i = 0; i < fn_ty->getNumParams(); i++)
      fn_args_ty.push_back(fn_ty->getParamType(i));
    fn_args_ty.push_back(builder_->getInt32Ty());
    fn_args_ty.push_back(builder_->getInt32Ty());
    fn_args_ty.push_back(builder_->getInt32Ty());
    fn_ty = FunctionType::get(fn_ret_ty, fn_args_ty, false);
  }
  Function *ret = Function::Create(fn_ty, Function::ExternalLinkage, fn->get_name(), mod_);
  // set attributes
  for(auto attr_pair: fn->attrs()){
    unsigned id = attr_pair.first;
    for(ir::attribute attr: attr_pair.second)
    if(attr.is_llvm_attr()){
      llvm::Attribute llattr = llvm_attr(ctx, attr);
      if(llattr.getKindAsEnum() != llvm::Attribute::None)
        ret->addAttribute(id, llvm_attr(ctx, attr));
    }
  }
  // set metadata
  if(tgt_->is_gpu()){
      tgt_->set_kernel(*builder_, ctx, mod_, ret);
      Metadata *md_args[] = {
        ValueAsMetadata::get(ret),
        MDString::get(ctx, "maxntidx"),
        ValueAsMetadata::get(builder_->getInt32(num_warps_*32))
      };
      mod_->getOrInsertNamedMetadata("nvvm.annotations")->addOperand(MDNode::get(ctx, md_args));
  }
  // set arguments
  for(unsigned i = 0; i < fn->args().size(); i++)
    vals_[fn->args()[i]][{}] = &*(ret->arg_begin() + i);
  // create blocks
  for(ir::basic_block *block: fn->blocks()) {
    BasicBlock *dst_block = BasicBlock::Create(ctx, block->get_name(), ret);
    bbs_[block] = dst_block;
  }
  builder_->SetInsertPoint(bbs_[fn->blocks()[0]]);
  // initialize layouts
  for(auto x: layouts_->get_all()){
    visit_layout(x.second);
  }
  // generate LLVM-IR code
  for(ir::basic_block *block: fn->blocks())
    visit_basic_block(block);
  // finalize
  finalize_function(fn);
}



void generator::visit_layout_hmma_884(analysis::mma_layout* layout) {
  ir::value *a = nullptr;
  ir::value *b = nullptr;
  for(ir::value* v: layout->get_values())
    if(ir::dot_inst* dot = dynamic_cast<ir::dot_inst*>(v)){
      a = dot->get_operand(0);
      b = dot->get_operand(1);
    }
  analysis::data_layout* layout_a = layouts_->get(a);
  analysis::data_layout* layout_b = layouts_->get(b);

  const auto& shape = layout->get_shape();
  Value *_1 = builder_->getInt32(1);
  Value *_2 = builder_->getInt32(2);
  Value *_3 = builder_->getInt32(3);
  Value *_4 = builder_->getInt32(4);
  Value *_8 = builder_->getInt32(8);
  Value *_16 = builder_->getInt32(16);
  Value *_32 = builder_->getInt32(32);
  int cc = tgt_->as_nvidia()->sm();
  std::vector<Value*> idx_m;
  std::vector<Value*> idx_n;
  std::vector<Value*> idx_z;
  //
  Value* thread = tgt_->get_local_id(mod_, *builder_, 0);
  Value *lane = builder_->CreateURem(thread, _32);
  Value *warp = builder_->CreateUDiv(thread, _32);
  /* lane offset */
  if(cc < 80){
    auto ord_a = layout_a->get_order();
    auto ord_b = layout_b->get_order();
    bool is_a_row = ord_a[0] != 0;
    bool is_b_row = ord_b[0] != 0;
    /* warp offset */
    Value *warp_0 = builder_->CreateURem(warp, builder_->getInt32(layout->wpt(0)));
    Value *warp_12 = builder_->CreateUDiv(warp, builder_->getInt32(layout->wpt(0)));
    Value *warp_1 = builder_->CreateURem(warp_12, builder_->getInt32(layout->wpt(1)));
    Value *off_warp_m = builder_->CreateMul(warp_0, builder_->getInt32(layout->spw(0)));
    Value *off_warp_n = builder_->CreateMul(warp_1, builder_->getInt32(layout->spw(1)));
    // Quad offset
    Value *off_quad_m = builder_->CreateMul(builder_->CreateUDiv(builder_->CreateAnd(lane, _16), _4), builder_->getInt32(layout->fpw(0)));
    Value *off_quad_n = builder_->CreateMul(builder_->CreateUDiv(builder_->CreateAnd(lane, _16), _4), builder_->getInt32(layout->fpw(1)));
    // Pair offset
    Value *off_pair_m = builder_->CreateUDiv(builder_->CreateURem(lane, _16), _4);
    off_pair_m = builder_->CreateURem(off_pair_m, builder_->getInt32(layout->fpw(0)));
    off_pair_m = builder_->CreateMul(off_pair_m, builder_->getInt32(4));
    Value *off_pair_n = builder_->CreateUDiv(builder_->CreateURem(lane, _16), _4);
    off_pair_n = builder_->CreateUDiv(off_pair_n, builder_->getInt32(layout->fpw(0)));
    off_pair_n = builder_->CreateURem(off_pair_n, builder_->getInt32(layout->fpw(1)));
    off_pair_n = builder_->CreateMul(off_pair_n, builder_->getInt32(4));
    // scale
    off_pair_m = builder_->CreateMul(off_pair_m, builder_->getInt32(layout->rep(0)/2));
    off_quad_m = builder_->CreateMul(off_quad_m, builder_->getInt32(layout->rep(0)/2));
    off_pair_n = builder_->CreateMul(off_pair_n, builder_->getInt32(layout->rep(1)/2));
    off_quad_n = builder_->CreateMul(off_quad_n, builder_->getInt32(layout->rep(1)/2));
    // Quad pair offset
    Value *off_lane_m = builder_->CreateAdd(off_pair_m, off_quad_m);
    Value *off_lane_n = builder_->CreateAdd(off_pair_n, off_quad_n);
    // a offset
    offset_a_m_[layout] = builder_->CreateAdd(off_warp_m, off_lane_m);
    offset_a_k_[layout] = builder_->CreateAnd(lane, _3);
    // b offsets
    offset_b_n_[layout] = builder_->CreateAdd(off_warp_n, off_lane_n);
    offset_b_k_[layout] = builder_->CreateAnd(lane, _3);
    // i indices
    Value *offset_c_m = builder_->CreateAdd(builder_->CreateAnd(lane, _1), offset_a_m_[layout]);
    for(unsigned m = 0; m < shape[0]; m+=layout->spt(0))
    for(unsigned mm = 0; mm < layout->rep(0); mm++)
      idx_m.push_back(builder_->CreateAdd(offset_c_m, builder_->getInt32(m + mm*2)));
    // j indices
    Value *offset_c_n = builder_->CreateAdd(builder_->CreateAnd(lane, _2), builder_->CreateAdd(off_warp_n, off_pair_n));
    for(unsigned n = 0; n < shape[1]; n+=layout->spt(1))
    for(unsigned nn = 0; nn < layout->rep(1); nn++){
      idx_n.push_back(builder_->CreateAdd(offset_c_n, builder_->getInt32(n + nn/2*4 + (nn%2)*2*layout->fpw(1)*layout->rep(1))));
      idx_n.push_back(builder_->CreateAdd(offset_c_n, builder_->getInt32(n + nn/2*4 + (nn%2)*2*layout->fpw(1)*layout->rep(1) + 1)));
    }
    if(is_a_row){
      offset_a_m_[layout] = builder_->CreateAdd(offset_a_m_[layout], builder_->CreateURem(thread, builder_->getInt32(4)));
      offset_a_k_[layout] = builder_->getInt32(0);
    }
    if(!is_b_row){
      offset_b_n_[layout] = builder_->CreateAdd(offset_b_n_[layout], builder_->CreateURem(thread, builder_->getInt32(4)));
      offset_b_k_[layout] = builder_->getInt32(0);
    }
    /* axes */
    axes_[layout->get_axis(0)] = distributed_axis{1, idx_m, warp_0};
    axes_[layout->get_axis(1)] = distributed_axis{1, idx_n, warp_1};
  }
  else{
    /* warp offset */
    Value *warp_0 = builder_->CreateURem(warp, builder_->getInt32(layout->wpt(0)));
    Value *warp_12 = builder_->CreateUDiv(warp, builder_->getInt32(layout->wpt(0)));
    Value *warp_1 = builder_->CreateURem(warp_12, builder_->getInt32(layout->wpt(1)));
    Value *off_warp_m = builder_->CreateMul(warp_0, builder_->getInt32(layout->spw(0)));
    Value *off_warp_n = builder_->CreateMul(warp_1, builder_->getInt32(layout->spw(1)));
    Value *off_lane_m = builder_->CreateURem(lane, _16);
    Value *off_lane_n = builder_->CreateURem(lane, _8);
    /* offsets */
    // a offset
    offset_a_m_[layout] = builder_->CreateAdd(off_warp_m, off_lane_m);
    offset_a_k_[layout] = builder_->getInt32(0);
    // b offsets
    offset_b_n_[layout] = builder_->CreateAdd(off_warp_n, off_lane_n);
    offset_b_k_[layout] = builder_->getInt32(0);
    // c offset
    Value *off_c_m = builder_->CreateAdd(builder_->CreateUDiv(lane, _4), off_warp_m);
    Value *off_c_n = builder_->CreateAdd(builder_->CreateMul(_2, builder_->CreateURem(lane, _4)), off_warp_n);
    for(unsigned m = 0; m < shape[0]; m+=layout->spt(0)){
      idx_m.push_back(builder_->CreateAdd(off_c_m, builder_->getInt32(m)));
      idx_m.push_back(builder_->CreateAdd(off_c_m, builder_->getInt32(m + 8)));
    }
    for(unsigned n = 0; n < shape[1]; n+=layout->spt(1)){
      idx_n.push_back(builder_->CreateAdd(off_c_n, builder_->getInt32(n)));
      idx_n.push_back(builder_->CreateAdd(off_c_n, builder_->getInt32(n + 1)));
    }
    /* axes */
    axes_[layout->get_axis(0)] = distributed_axis{1, idx_m, warp_0};
    axes_[layout->get_axis(1)] = distributed_axis{1, idx_n, warp_1};
  }
}

void generator::visit_layout_scanline(analysis::scanline_layout* layout) {
  Value *warp_size = builder_->getInt32(32);
  Value* u_thread_id_0 = tgt_->get_local_id(mod_, *builder_, 0);
  Value *u_thread_id = builder_->CreateURem(u_thread_id_0, warp_size);
  Value *u_warp_id = builder_->CreateUDiv(u_thread_id_0, warp_size);

  auto order = layout->get_order();
  const auto& shape = layout->get_shape();
  Value* full_thread_id = builder_->CreateAdd(builder_->CreateMul(u_warp_id, builder_->getInt32(32)), u_thread_id);
  // Delinearize
  size_t dim = shape.size();
  std::vector<Value*> thread_id(dim);
  for(unsigned k = 0; k < dim - 1; k++){
    Constant *dim_k = builder_->getInt32(layout->mts(order[k]));
    Value *rem = builder_->CreateURem(full_thread_id, dim_k);
    full_thread_id = builder_->CreateUDiv(full_thread_id, dim_k);
    thread_id[order[k]] = rem;
  }
  thread_id[order[dim - 1]] = full_thread_id;
  // Create axes
  for(unsigned k = 0; k < dim; k++) {
    int nts = layout->nts(k);
    int mts = layout->mts(k);
    std::string str_k = std::to_string(k);
    Value *contiguous_k = builder_->getInt32(nts);
    Value *scaled_thread_id = builder_->CreateMul(thread_id[k], contiguous_k);
    unsigned per_block  = nts * mts;
    unsigned per_thread = nts * shape[k] / per_block;
    std::vector<Value*> idx_list(per_thread);
    for(unsigned n = 0 ; n < per_thread; n++){
      unsigned offset = n / nts * per_block + n % nts;
      idx_list[n] = builder_->CreateAdd(scaled_thread_id, builder_->getInt32(offset), "idx_" + str_k + "_" + std::to_string(n));
    }
    axes_[layout->get_axis(k)] = distributed_axis{nts, idx_list, thread_id[k]};
  }
}

void generator::visit_layout_shared(analysis::shared_layout* layout) {
  Type* ty = llvm_type(layout->get_type(), builder_->getContext());
  PointerType *ptr_ty = ty->getPointerTo(shmem_->getType()->getPointerAddressSpace());
  // double-buffered
  if(layout->get_double_buffer()) {
    BasicBlock *current = builder_->GetInsertBlock();
    auto info = *layout->get_double_buffer();
    ir::phi_node *phi = info.phi;
    BasicBlock *parent = bbs_.at(phi->get_parent());
    if(parent->empty())
      builder_->SetInsertPoint(parent);
    else
      builder_->SetInsertPoint(&*parent->getFirstNonPHI());
    // create pointers
    shared_ptr_[layout] = builder_->CreatePHI(ptr_ty, 2);
    shared_pre_ptr_[layout] = builder_->CreateGEP(shmem_, builder_->getInt32(alloc_->offset(layout)));
    shared_pre_ptr_[layout] = builder_->CreateBitCast(shared_pre_ptr_[layout], shared_ptr_[layout]->getType());
    shared_off_[layout] = builder_->CreatePHI(builder_->getInt32Ty(), 2);
    shared_next_ptr_[layout] = builder_->CreateGEP(shared_ptr_[layout], shared_off_[layout], "next_ptr");
    builder_->SetInsertPoint(current);
  }
  else{
    size_t offset = alloc_->offset(layout);
    shared_ptr_[layout] = builder_->CreateGEP(shmem_, builder_->getInt32(offset));
    shared_ptr_[layout] = builder_->CreateBitCast(shared_ptr_[layout], ptr_ty);
  }
}

void generator::visit_basic_block(ir::basic_block * block) {
  BasicBlock *parent = bbs_[block];
  builder_->SetInsertPoint(parent);
  for(ir::instruction *i: block->get_inst_list()){
    visit_value(i);
  }
  bbs_[block] = builder_->GetInsertBlock();
}

void generator::visit_argument(ir::argument* arg) {

}

void generator::init_idx(ir::value *v) {
  idxs_[v].clear();
  if(!v->get_type()->is_tile_ty()){
    idxs_[v].push_back({});
    return;
  }
  if(layouts_->get(v)->to_shared())
    return;
  const auto &shapes = v->get_type()->get_tile_shapes();
  size_t rank = shapes.size();
  std::vector<distributed_axis> axes(rank);
  std::vector<int> ord(rank);
  // compute axes
  for(size_t d = 0; d < shapes.size(); d++){
    if(shapes[d] > 1){
      unsigned x = a_axes_->get(v, d);
      axes[d] = axes_.at(x);
    }
    else{
      axes[d].contiguous = 1;
      axes[d].values = {builder_->getInt32(0)};
    }
  }
  // compute order
  analysis::data_layout* layout = layouts_->get(v);
  std::iota(ord.begin(), ord.end(), 0);
  auto cmp = [&](int x, int y) {
    unsigned axx = a_axes_->get(v, x);
    unsigned axy = a_axes_->get(v, y);
    size_t posx = layout->find_axis(axx);
    size_t posy = layout->find_axis(axy);
    if(posx < rank && posy < rank)
      return layout->get_order(posx) < layout->get_order(posy);
    return false;
  };
  std::sort(ord.begin(), ord.end(), cmp);
  // indices
  if(axes.size() == 1)
    for(Value* x0: axes[ord[0]].values){
      idxs_[v].push_back({x0});
    }
  if(axes.size() == 2)
    for(Value* x1: axes[ord[1]].values)
    for(Value* x0: axes[ord[0]].values){
      indices_t idx(2);
      idx[ord[0]] = x0;
      idx[ord[1]] = x1;
      idxs_[v].push_back(idx);
    }
}

void generator::finalize_shared_layout(analysis::shared_layout *shared) {
  if(shared->get_double_buffer()) {
    auto info = *shared->get_double_buffer();
    ir::phi_node *phi = info.phi;
    PHINode *ptr = (PHINode*)shmems_[phi];
    PHINode *offset = (PHINode*)shoffs_[phi];
    for(unsigned n = 0; n < phi->get_num_incoming(); n++){
      ir::basic_block* inc_block = phi->get_incoming_block(n);
      ir::value* inc_val = phi->get_incoming_value(n);
      BasicBlock *llvm_inc_block = bbs_.at(inc_block);
      if(inc_val == info.latch){
        builder_->SetInsertPoint(llvm_inc_block->getTerminator());
        Value *next_offset = builder_->CreateNeg(offset);
        offset->addIncoming(next_offset, llvm_inc_block);
      }
      else {
        unsigned num_bytes = shared->get_type()->get_primitive_size_in_bits() / 8;
        offset->addIncoming(builder_->getInt32(shared->get_size() / (2*num_bytes)), llvm_inc_block);
      }
      ptr->addIncoming(shmems_[inc_val], llvm_inc_block);
    }
  }
}

void generator::finalize_function(ir::function *fn) {
  // finalize double-buffering
  for(const auto& x: layouts_->get_all())
  if(auto *shared = dynamic_cast<analysis::shared_layout*>(x.second))
    finalize_shared_layout(shared);
  // finalize phi
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *inst: block->get_inst_list())
    if(auto *phi = dynamic_cast<ir::phi_node*>(inst))
      finalize_phi_node(phi);
}

void generator::finalize_phi_node(ir::phi_node *x) {
  if(shmems_.find(x) != shmems_.end())
    return;
  for(unsigned n = 0; n < x->get_num_incoming(); n++){
    ir::basic_block *_block = x->get_incoming_block(n);
    BasicBlock *block = bbs_.at(_block);
    for(indices_t idx: idxs_.at(x)){
      PHINode *phi = (PHINode*)vals_[x][idx];
      Value *inc = vals_[x->get_incoming_value(n)][idx];
      phi->addIncoming(inc, block);
    }
  }
}

void generator::visit(ir::module &src, llvm::Module &dst) {
  mod_ = &dst;
  ctx_ = &dst.getContext();
  builder_ = new Builder(*ctx_);
  // allocate shared memory
  if(tgt_->is_gpu())
  if(unsigned alloc_size = alloc_->allocated_size()){
    Type *int_8_ty = Type::getInt8Ty(*ctx_);
    Type *int_32_ty = Type::getInt32Ty(*ctx_);
    ArrayType *array_ty = ArrayType::get(int_32_ty, alloc_size/4);
    Type *ptr_ty = PointerType::get(int_8_ty, 3);
    GlobalVariable *sh_mem_array =
      new GlobalVariable(*mod_, array_ty, false, GlobalVariable::ExternalWeakLinkage,
                         nullptr, "__shared_ptr", nullptr, GlobalVariable::NotThreadLocal, 3);
    shmem_ = builder_->CreateBitCast(sh_mem_array, ptr_ty);
  }
  // visit functions
  for(ir::function *fn: src.get_function_list())
    visit_function(fn);
}


}
}
