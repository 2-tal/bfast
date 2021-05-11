import "lib/github.com/diku-dk/sorts/insertion_sort"
import "helpers"

-- | implementation is in this entry point
--   the outer map is distributed directly
let mainFun [m][N] (trend: i32) (k: i32) (n: i32) (freq: f32)
                  (hfrac: f32) (lam: f32)
                  (mappingindices : [N]i32)
                  (images : [m][N]f32) (hist_start: i64) =
  ----------------------------------
  -- 1. make interpolation matrix --
  ----------------------------------
  let n64 = i64.i32 n
  let k2p2 = 2 * k + 2
  let k2p2' = if trend > 0 then k2p2 else k2p2-1
  let X = (if trend > 0
           then mkX_with_trend (i64.i32 k2p2') freq mappingindices
           else mkX_no_trend (i64.i32 k2p2') freq mappingindices)
          |> intrinsics.opaque

  -- PERFORMANCE BUG: instead of `let Xt = copy (transpose X)`
  --   we need to write the following ugly thing to force manifestation:
  let zero = r32 <| i32.i64 <| (N * N + 2 * N + 1) / (N + 1) - N - 1
  let Xt  = intrinsics.opaque <| map (map (+zero)) (copy (transpose X))

  let Xh  = X[:,:n64]
  let Xth = Xt[:n64,:]
  let Yh  = images[:,:n64]

  -- [ ] Cancel their fit and use the one from recursive residuals at
  -- break time? Not possible bc. beta is updated iteratively and
  -- we have no way to know break at that time.
  --
  -- Xh is only used for fitting history (kernels 1,2,4).
  -- Xth is only used for Xsqr (kernel 2).
  -- Yh is used for fitting history (kernel 2,4)
  --    and to compute number of hist. non-nans (kernel 6).
  --
  -- Yh is used as filter for Xsqr and later; seems
  -- like I can literally just set non-stable to nans in Yh only?

  let stable_history_starts = replicate m hist_start

  -- Filter out unstable values!
  let Yh = map2 (\i y ->
                   -- TODO hack bc python operates on non-nan versions;
                   --      map stable hist idx to non-nan idx
                   let num_nn = map (\yj -> if f32.isnan yj then 0i64 else 1i64) y
                                |> scan (+) 0i64 
                   in map2 (\j_nn yj -> if j_nn-1 < i then f32.nan else yj) num_nn y
                ) stable_history_starts Yh

  ----------------------------------
  -- 2. mat-mat multiplication    --
  ----------------------------------
  let Xsqr = intrinsics.opaque <| map (matmul_filt Xh Xth) Yh

  ----------------------------------
  -- 3. matrix inversion          --
  ----------------------------------
  let Xinv = intrinsics.opaque <| map mat_inv Xsqr

  ---------------------------------------------
  -- 4. several matrix-vector multiplication --
  ---------------------------------------------
  let beta0  = map (matvecmul_row_filt Xh) Yh   -- [m][2k+2]
               |> intrinsics.opaque

  let beta   = map2 matvecmul_row Xinv beta0    -- [m][2k+2]
               |> intrinsics.opaque -- ^ requires transposition of Xinv
                                    --   unless all parallelism is exploited

  let y_preds= map (matvecmul_row Xt) beta      -- [m][N]
               |> intrinsics.opaque -- ^ requires transposition of Xt (small)
                                    --   can be eliminated by passing
                                    --   (transpose X) instead of Xt
  -- TODO ^ currently here, y_preds do not match once
  --        hist start > 0 !!
  in y_preds

let y =
  [4064f32,f32.nan,f32.nan,4628f32,4034f32,f32.nan,3867f32,2117f32,4598f32,f32.nan,f32.nan,f32.nan,
   f32.nan,4210f32,f32.nan,5253f32,4219f32,f32.nan,3074f32,f32.nan,f32.nan,3982f32,f32.nan,f32.nan,
   f32.nan,f32.nan,3888f32,f32.nan,f32.nan,f32.nan,3662f32,4182f32,4475f32,4659f32,5102f32,4552f32,
   f32.nan,f32.nan,f32.nan,f32.nan,3827f32,f32.nan,f32.nan,f32.nan,3773f32,3985f32,f32.nan,f32.nan,
   5331f32,f32.nan,f32.nan,f32.nan,f32.nan,f32.nan,4165f32,f32.nan,3965f32,f32.nan,4094f32,f32.nan,
   f32.nan,f32.nan,f32.nan,3778f32,f32.nan,3176f32,4073f32,4461f32,4017f32,4031f32,3929f32,3999f32,
   f32.nan,4349f32,4182f32,f32.nan,f32.nan,f32.nan,3001f32,3286f32,f32.nan,f32.nan,f32.nan,f32.nan,
   f32.nan,4212f32,4309f32,f32.nan,4451f32,f32.nan,f32.nan,3999f32,4064f32,f32.nan,f32.nan,4215f32,
   f32.nan,f32.nan,f32.nan,3917f32,4035f32,4263f32,4286f32,f32.nan,f32.nan,4062f32,f32.nan,f32.nan,
   f32.nan,f32.nan,f32.nan,4129f32,4042f32,f32.nan,3923f32,f32.nan,4407f32,f32.nan,f32.nan,f32.nan,
   f32.nan,f32.nan,4716f32,f32.nan,f32.nan,f32.nan,f32.nan,f32.nan,f32.nan,f32.nan,4690f32,4570f32,
   4505f32,f32.nan,4172f32,f32.nan,f32.nan,f32.nan,f32.nan,4225f32,f32.nan,f32.nan,f32.nan,4402f32,
   4168f32,4053f32,f32.nan,f32.nan,f32.nan,f32.nan,4237f32,f32.nan,f32.nan,f32.nan,4035f32,3533f32,
   f32.nan,f32.nan,f32.nan,f32.nan,4050f32,4341f32,f32.nan,4555f32,4096f32,f32.nan,4077f32,4196f32,
   f32.nan,f32.nan,4205f32,4647f32,f32.nan,4005f32,3607f32,f32.nan,f32.nan,f32.nan,f32.nan,3977f32,
   3761f32,f32.nan,f32.nan,f32.nan,4184f32,2652f32,4341f32,f32.nan,f32.nan,4476f32,4257f32,f32.nan,
   4066f32,f32.nan,f32.nan,f32.nan,4836f32,f32.nan,f32.nan,4048f32,f32.nan,f32.nan,f32.nan,f32.nan,
   f32.nan,4533f32,4330f32,f32.nan,4554f32,4100f32,f32.nan,3692f32,f32.nan,4337f32,f32.nan,f32.nan]

let mapped_indices =
[194i32,210,226,242,258,274,290,306,386,498,546,578,610,626
,706,722,754,802,898,914,946,978,1050,1058,1098,1138,1178,1226
,1234,1242,1250,1282,1314,1322,1330,1346,1354,1362,1378,1394,1490,1498
,1554,1562,1586,1594,1602,1610,1626,1634,1642,1658,1666,1674,1682,1698
,1714,1722,1730,1738,1802,1834,1938,1954,1962,1970,1978,1986,2010,2026
,2034,2050,2058,2082,2098,2146,2170,2178,2258,2274,2322,2330,2362,2370
,2402,2418,2426,2434,2442,2466,2506,2514,2530,2554,2682,2690,2714,2722
,2778,2786,2794,2810,2826,2842,2850,2866,2874,2898,2906,2994,3042,3114
,3122,3146,3154,3162,3170,3178,3186,3194,3210,3258,3338,3370,3386,3418
,3434,3450,3466,3482,3498,3514,3530,3546,3562,3626,3642,3762,3770,3778
,3794,3826,3898,3906,3922,3930,3946,3954,3962,3986,4010,4018,4034,4042
,4066,4106,4114,4138,4154,4186,4194,4202,4210,4226,4266,4282,4290,4314
,4322,4330,4346,4354,4362,4370,4418,4434,4466,4522,4530,4546,4554,4562
,4570,4586,4594,4602,4610,4618,4634,4642,4650,4658,4666,4690,4706,4714
,4722,4762,4770,4778,4818,4850,4874,4882,4898,4914,4930,4938,4946,4954
,4970,4978,4986,4994,5002,5018]

let main = mainFun 1 3 94 365 0.25 1.897626420474509f32 mapped_indices [y]
