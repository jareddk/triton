import itertools
import torch
import triton as tt
import pytest

def dense_to_sparse(w, mask, block):
  Z = w.size(0)
  ret = torch.empty((Z, mask.sum(), block, block), dtype=w.dtype, device=w.device)
  nnz = mask.nonzero(as_tuple=False)
  h, i, j = nnz[:, 0], nnz[:, 1], nnz[:, 2]
  for zz in range(Z):
    for idx, (hh, ii, jj) in enumerate(zip(h, i, j)):
      ret[zz, idx, :, :] = w[zz, hh, ii*block: (ii+1)*block, jj*block: (jj+1)*block]
  return ret

def sparse_to_dense(w, mask, block, zero = 0):
  maskedw = w.clone()
  for bz, wz in enumerate(range(0, w.size(0))):
    for bh, wh in enumerate(range(0, w.size(1))):
      for bi, wi in enumerate(range(0, w.size(2), block)):
        for bj, wj in enumerate(range(0, w.size(3), block)):
          if mask[bh, bi, bj] == 0:
            maskedw[wz, wh, wi : wi+block, wj:wj+block] = zero
  return maskedw

@pytest.mark.parametrize("MODE, TRANS_A, TRANS_B, BLOCK", 
    [
    (mode, at, bt, block) for mode in ['sdd', 'dsd', 'dds']\
                          for at   in [False, True]\
                          for bt   in [False, True]\
                          for block in [16, 32, 64]
    ]
)
def test_op(MODE, TRANS_A, TRANS_B, BLOCK, DTYPE = torch.float16, Z = 3, H = 2, M = 128, N = 256, K = 384):
  # set seed
  torch.random.manual_seed(0)
  # create inputs
  x = torch.randn((Z, H, K, M) if TRANS_A else (Z, H, M, K), dtype=DTYPE, device='cuda')
  w = torch.randn((Z, H, N, K) if TRANS_B else (Z, H, K, N), dtype=DTYPE, device='cuda')
  shape = {'sdd': (M, N), 'dsd': (x.shape[2], x.shape[3]), 'dds': (w.shape[2], w.shape[3])}[MODE]
  layout = torch.randint(2, (H, shape[0]//BLOCK, shape[1]//BLOCK))
  # triton result
  lhs = dense_to_sparse(x, layout, BLOCK) if MODE == 'dsd' else x
  rhs = dense_to_sparse(w, layout, BLOCK) if MODE == 'dds' else w
  op = tt.ops.blocksparse_matmul(layout, BLOCK, MODE, trans_a=TRANS_A, trans_b=TRANS_B)
  ry  = op(lhs, rhs)
  # torch result
  lhs = sparse_to_dense(x, layout, BLOCK) if MODE == 'dsd' else x
  rhs = sparse_to_dense(w, layout, BLOCK) if MODE == 'dds' else w
  lhs = lhs.transpose(2, 3) if TRANS_A else lhs
  rhs = rhs.transpose(2, 3) if TRANS_B else rhs
  ty = torch.matmul(lhs, rhs)
  ty = sparse_to_dense(ty, layout, BLOCK) if MODE == 'sdd' else ty
  ty = dense_to_sparse(ty, layout, BLOCK) if MODE == 'sdd' else ty
  # compare
  rtol, atol = {torch.float32: (1e-4, 1e-5),
                torch.float16: (1e-2, 1e-3)}[DTYPE]
  assert torch.allclose(ry, ty, rtol=rtol, atol=atol)
