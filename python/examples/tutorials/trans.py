import torch
import triton

class _transpose(torch.autograd.Function):
    src = """
    __global__ void transpose(TYPE * X, TYPE * Y,
                              int M __retune,
                              int N __retune,
                              int ldx __multipleof(8), int ldy __multipleof(8)) {
          // extract program ID
          int pidm = get_program_id(0); //(1)
          int pidn = get_program_id(1); //(2)

          // create 1D range along the two matrix's axes
          int rm[TM] = pidm * TM + 0 ... TM; //(3)
          int rn[TN] = pidn * TN + 0 ... TN; //(4)

          // create 2D array of pointers
          TYPE* px[TM, TN] = X + rm[:, newaxis] * ldx + rn[newaxis, :]; //(5)
          TYPE* py[TN, TM] = Y + rm[newaxis, :] + rn[:, newaxis] * ldy; //(6)

          // create bounds-checking mask
          bool checkx[TM, TN] = (rm[:, newaxis] < M) && (rn[newaxis, :] < N); //(7a)
          bool checky[TN, TM] = (rn[:, newaxis] < N) && (rm[newaxis, :] < M); //(7b)

          // conditional write-back using the conditional dereferencing operatior '*?()'
          *?(checky)py = ^(*?(checkx)px); //(7)
        }
    """

    kernel = None ### initialize later when we know the sizes

    @staticmethod
    def forward(ctx, x):

       M, N = x.shape

       ldx = N
       ldy = M

       dtype = x.dtype

       y = torch.empty((N,M)).cuda()

       defines= {
           'TYPE' : dtype,
           'TM'   : [32,64,128],
           'TN'   : [32,64,128],
       }

       grid = lambda opt: [triton.cdiv(M, opt.d('TM')), triton.cdiv(N, opt.d('TN'))]

       if _transpose.kernel is None:
           _transpose.kernel = triton.kernel(_transpose.src, defines=defines, num_warps=[4])

       _transpose.kernel(x, y, M, N, ldx, ldy, grid=grid)

       return y

transpose = _transpose.apply

# test
torch.manual_seed(0)
x = torch.randn(1024,128).cuda()

print(x)

ya = torch.t(x)
yb = transpose(x)
print()
print(ya)
print()
print(yb)
print(torch.allclose(ya, yb))

print(ya == yb)
