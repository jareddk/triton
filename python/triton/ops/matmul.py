import torch
import triton

class _matmul(torch.autograd.Function):
    src = """
#define STM 8
#define STN 8

__global__ void matmul(TYPE * A __noalias __readonly __aligned(16),
                       TYPE * B __noalias __readonly __aligned(16),
                       TYPE * C __noalias __aligned(16),
                       float alpha,
                       int M __retune,
                       int N __retune,
                       int K __retune __multipleof(16),
                       int stride_am __multipleof(M_STRIDE_AM), int stride_ak __multipleof(M_STRIDE_AK),
                       int stride_bk __multipleof(M_STRIDE_BK), int stride_bn __multipleof(M_STRIDE_BN),
                       int stride_cm __multipleof(M_STRIDE_CM), int stride_cn __multipleof(M_STRIDE_CN),
                       int* locks) {
      // prologue
      int pid = get_program_id(0);
      int pidz = get_program_id(2);
      int gridm = (M + TM - 1) / TM;
      int gridn = (N + TN - 1) / TN;

      // swizzle for better L2 performance
      int width = STM*gridn;
      int stm = pid / width;
      int RSTM  = min(gridm - stm*STM, STM);
      int stn =  (pid % width) / (RSTM*STN);
      int RSTN = min(gridn - stn*STN, STN);
      int laneid = pid % (RSTM * RSTN);
      int lanem = laneid / RSTN;
      int lanen = laneid % RSTN;
      int pidm = stm*STM + lanem;
      int pidn = stn*STN + lanen;
      int rm[TM] = pidm * TM + 0 ... TM;
      int rn[TN] = pidn * TN + 0 ... TN;

      // split-k for better parrelism
      K           = K / TZ;
      int rk[TK]  = 0 ... TK;
      // pointers to operands
      int offa[TM, TK] = (pidz*K + rk[newaxis, :]) * STRIDE_AK + rm[:, newaxis] * STRIDE_AM;
      int offb[TK, TN] = (pidz*K + rk[:, newaxis]) * STRIDE_BK + rn[newaxis, :] * STRIDE_BN;
      TYPE* pa[TM, TK] = A + offa;
      TYPE* pb[TK, TN] = B + offb;

      // prefetches operands
      bool checka[TM, TK] = rk[newaxis, :] < K;
      bool checkb[TK, TN] = rk[:, newaxis] < K;
      TYPE a[TM, TK] = checka ? *pa : 0;
      TYPE b[TK, TN] = checkb ? *pb : 0;
      pa += TK * STRIDE_AK;
      pb += TK * STRIDE_BK;

      // reduction loop
      float acc[TM, TN] = 0;
      for(int k = K; k > 0; k -= TK){
#ifdef K_MULTIPLE_OF_TK
        bool checkk[TK] = k > TK;
#else
        bool checkk[TK] = rk < k - TK;
#endif
        bool checka[TM, TK] = checkk[newaxis, :];
        bool checkb[TK, TN] = checkk[:, newaxis];
        acc += a @ b;
#ifdef K_MULTIPLE_OF_TK
        a = *?(checka)pa;
        b = *?(checkb)pb;
#else
        a = checka ? *pa : 0;
        b = checkb ? *pb : 0;
#endif
        pa += TK * STRIDE_AK;
        pb += TK * STRIDE_BK;
      }
      acc = acc * alpha;
      TYPE c[TM, TN] = acc;

      // epilogue
      int rcm[TM] = pidm * TM + 0 ... TM;
      int rcn[TN] = pidn * TN + 0 ... TN;
      int offc[TM, TN] = rcm[:, newaxis] * stride_cm + rcn[newaxis, :];
      TYPE* pc[TM, TN] = C + offc;
      bool checkc[TM, TN] = rcm[:, newaxis] < M && rcn[newaxis, :] < N;
#if (TZ==1)
      *?(checkc) pc = c;
#else
      // accumulate partial result using spin-locks
      int *plock  = locks + rid;
      int *pcount = plock + get_num_programs(0) * get_num_programs(1);
      for(int repeat = 1; repeat == 1; repeat = atomic_cas(plock, 0, 1));
      int count = *pcount;
      if(count == 0)
        *?(checkc) pc = c;
      else
        *?(checkc) pc = c + *?(checkc)pc;
      atomic_xchg(pcount, (count + 1) % TZ);
      atomic_xchg(plock, 0);
#endif
}
    """
    TM = 128
    TN = 128
    TK = 32
    TZ = 1
    num_warps = 4
    kernel = dict()

    @staticmethod
    def multiple_of(N):
        if N % 8 == 0: return 8
        if N % 4 == 0: return 4
        if N % 2 == 0: return 2
        return 1

        
    _locks = dict()
    @staticmethod
    def get_locks(dev):
        if dev not in _matmul._locks:
            _matmul._locks[dev] = torch.zeros(1024*1024, dtype=torch.int32, device=dev)
        return _matmul._locks[dev]

    @staticmethod
    def _call(a, b):
        # allocate output
        M, K = a.shape
        K, N = b.shape
        c = torch.empty((M, N), dtype=a.dtype, device=a.device)
        # kernel hash
        m_stride_am = _matmul.multiple_of(a.stride(0))
        m_stride_ak = _matmul.multiple_of(a.stride(1))
        m_stride_bk = _matmul.multiple_of(b.stride(0))
        m_stride_bn = _matmul.multiple_of(b.stride(1))
        m_stride_cm = _matmul.multiple_of(c.stride(0))
        m_stride_cn = _matmul.multiple_of(c.stride(1))
        m_k_16      = K % 16 == 0
        key = (c.device, c.dtype, a.stride(0) == 1, a.stride(1) == 1, b.stride(0) == 1, b.stride(1) == 1,
               m_stride_am, m_stride_ak, m_stride_bk, m_stride_bn, m_stride_cm, m_stride_cn, m_k_16)
        if key not in _matmul.kernel:
            defines = {
                'TYPE' : c.dtype,
                'SHAPE_A': 'TM, TK', 'SHAPE_B': 'TK, TN',
                'STRIDE_AM': '1' if a.stride(0) == 1 else 'stride_am', 
                'STRIDE_AK': '1' if a.stride(1) == 1 else 'stride_ak',
                'STRIDE_BK': '1' if b.stride(0) == 1 else 'stride_bk',
                'STRIDE_BN': '1' if b.stride(1) == 1 else 'stride_bn',
                'M_STRIDE_AM': m_stride_am,
                'M_STRIDE_AK': m_stride_ak,
                'M_STRIDE_BK': m_stride_bk,
                'M_STRIDE_BN': m_stride_bn,
                'M_STRIDE_CM': m_stride_cm,
                'M_STRIDE_CN': m_stride_cn,
                'TM'   : _matmul.TM,
                'TN'   : _matmul.TN,
                'TK'   : _matmul.TK,
                'TZ'   : _matmul.TZ
            }
            if m_k_16:
                defines['K_MULTIPLE_OF_TK'] = '1'
            _matmul.kernel[key] = triton.kernel(_matmul.src, num_warps=_matmul.num_warps, defines=defines)
        kernel = _matmul.kernel[key]
        # enqueue
        kernel(a, b, c, 1., M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), _matmul.get_locks(c.device), 
               grid = lambda opt: [triton.cdiv(M, opt.d('TM'))*triton.cdiv(N, opt.d('TN'))])
        return c

    @staticmethod
    def forward(ctx, a, b):
        c = _matmul._call(a,b)
        return c

matmul = _matmul.apply
