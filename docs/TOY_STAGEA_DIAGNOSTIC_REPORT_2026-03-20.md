# Toy Stage-A Diagnostic Report (2026-03-20)

## 1) Executive Summary
- Mục tiêu kiểm chứng: phương pháp trajectory attack có hỗ trợ claim trong `draft_my_method.md` dưới **limited-data setting** hay không.
- Kết luận hiện tại: **chưa đủ bằng chứng để xác nhận theory một cách ổn định**.
- Cái đã xác nhận:
  - Cơ chế attack hoạt động (không còn collapse giả sau khi fix bug).
  - Có seed cho thấy robust thắng baseline trên attacked heldout mà vẫn giữ clean/sample.
- Cái chưa đạt:
  - Kết quả chưa ổn định qua seed (seed sensitivity cao), nên chưa thể claim theory được support nhất quán.

## 2) Scope Và Protocol Fair Compare
- Codebase: `Wild-Diffusion/toy`.
- Setting: **limited-data thật** (fixed finite train pool), không phải population sampling.
- Dataset toy:
  - 8-mode circle GMM.
  - `train_points_per_mode=2` => train pool = 16 điểm cố định.
  - heldout pool = 10,000 samples.
- Training/eval chung cho tất cả run chính:
  - `steps=3000`, `batch_size=1024`, `eval_samples=6000`.
  - `outer_clean_weight=1.0`, `outer_attack_weight=0.5`.
  - Warmup chuẩn: `warmup_clean_steps=900`, `warmup_ramp_steps=600`.
  - Gate đánh giá theory:
    - Attack cải thiện: robust_on_forward_attack tốt hơn baseline ở đa số timestep.
    - Clean không phá mạnh: robust_on_forward_baseline không tăng quá nhiều.
    - Sample quality không xấu rõ: coverage giữ 1.0, `avg/p90` không xấu mạnh.

## 3) Root Cause Quan Trọng Đã Fix
### 3.1 Bug
- Trong warmup, code cũ zero toàn bộ tham số `ControlNet`.
- Với kiến trúc `Linear -> Tanh -> Linear -> Tanh -> Linear`, zero-init gây nghẽn gradient ở hidden layers.

### 3.2 Bằng chứng gradient starvation
- Test local (đã chạy):
  - `zero_init`: toàn bộ gradient hidden layers = 0, chỉ `net.4.bias` có gradient.
  - `random_init`: tất cả layers đều có gradient bình thường.

### 3.3 Patch
- Đã bỏ zero-init trong [`trainer.py`](/Users/quan238/personal/VinUniversity/Wild-Diffusion/toy/trainer.py#L143).
- Checksum local/server đã đồng nhất trong lúc chạy thử nghiệm.

## 4) Diagnostic Runs (trước và sau fix)

### 4.1 Run `6254` (warmup trước fix)
- Exp: `baseline_attack_clean_anchor_warmup_diag_2ppm_logn_3k`.
- Kết quả:
  - `collapse_suspected=True`
  - `diag_v_l2_mean_last=0.0113`
  - `diag_gap_ratio_mean_last=-0.00419`
  - `robust_energy_mean_last=2.03e-4`
- Diễn giải: attack gần như no-op, đây là collapse giả do bug init.

### 4.2 Run `6255` (no warmup, trước fix)
- Exp: `baseline_attack_clean_anchor_diag_2ppm_logn_3k`.
- Kết quả:
  - `collapse_suspected=False`
  - `diag_v_l2_mean_last=6.3442`
  - `attack wins=19/24`, `att_gap_mean=-1.1258`
  - `clean_ratio=+61.5%`
  - sample gap: `avg_delta=+0.1036`, `p90_delta=+0.3237`
- Diễn giải: attack mạnh, robust trên attacked tăng rõ, nhưng clean/sample quality xấu nhiều.

### 4.3 Run `6259` (warmup sau fix zero-init)
- Exp: `baseline_attack_clean_anchor_warmup_diag_fixzero_2ppm_logn_3k`.
- Kết quả:
  - `collapse_suspected=False`
  - `diag_v_l2_mean_last=3.6713`
  - `attack wins=20/24`, `att_gap_mean=-1.0900`
  - `clean_ratio=+68.1%`
  - sample gap: `avg_delta=+0.1135`, `p90_delta=+0.3371`
- Diễn giải: warmup đã có attack thật (không collapse), nhưng trade-off clean/sample vẫn nặng.

## 5) Main Theory Check Sweep (Job `6260`)
- 9 runs: `lambda_energy ∈ {0.2,0.3,0.4} × seed ∈ {0,1,2}`.
- Exp names: `theory_check_lam{lam}_seed{seed}_2ppm_3k`.
- Tất cả run đều `collapse_suspected=False`.

### 5.1 Per-lambda tổng hợp
- `lambda=0.2`
  - `attack_wins_mean=17.00/24`
  - `att_gap_mean=-0.9348`
  - `clean_ratio_mean=+56.7%`
  - `avg_delta_mean=+0.0957`, `p90_delta_mean=+0.2948`
  - `overall_pass=0/3`
- `lambda=0.3`
  - `attack_wins_mean=13.00/24`
  - `att_gap_mean=-0.0095`
  - `clean_ratio_mean=+1.4%`
  - `avg_delta_mean=-0.0026`, `p90_delta_mean=-0.0066`
  - `overall_pass=1/3`
- `lambda=0.4`
  - `attack_wins_mean=10.67/24`
  - `att_gap_mean=-0.0018`
  - `clean_ratio_mean=+0.5%`
  - `avg_delta_mean=-0.0038`, `p90_delta_mean=-0.0097`
  - `overall_pass=1/3`

### 5.2 Diễn giải khoa học
- `lambda` thấp (`0.2`): attack đủ mạnh nhưng phá fidelity mạnh.
- `lambda` cao (`0.3-0.4`): clean/sample đẹp hơn nhưng gain attacked rất nhỏ và không ổn định qua seed.
- Pattern hiện tại: **không có vùng hyperparameter cho kết quả đồng thời tốt và ổn định qua seed**.

## 6) Verdict So Với Claim Theory
- Claim cần support: trong limited data, trajectory perturbation giúp robust hơn mà không phá mạnh generative fidelity.
- Trạng thái hiện tại:
  - **Mechanism check: PASS** (attack có hiệu lực, robust branch học được trên attacked states ở nhiều run).
  - **Theory-aligned acceptance: FAIL (chưa ổn định)**.
- Lý do fail:
  - Seed sensitivity cao.
  - Trade-off robustness vs fidelity chưa cân bằng ổn định.

## 7) Rủi Ro Khi Báo Kết Quả
- Không nên claim “theory confirmed”.
- Claim an toàn: “hiện tại mới xác nhận được mechanism; cần thêm kiểm định ổn định để support theory-level conclusion”.

## 8) Đề xuất 1 bước ưu tiên tiếp theo
- Ưu tiên duy nhất: **sweep 1 chiều `lambda_energy` mịn hơn quanh vùng trung gian (0.28–0.38) với nhiều seed hơn**, giữ nguyên protocol.
- Mục tiêu: tìm vùng ổn định đạt đồng thời 3 gate (attack/clean/sample) qua seed.

## 9) Artifacts Và Đường Dẫn
- Report tóm tắt sweep đã sync local:
  - [theory_lambda_sweep_summary.txt](/Users/quan238/personal/VinUniversity/toy_outputs/theory_lambda_sweep_summary.txt)
- Code patch chính:
  - [trainer.py](/Users/quan238/personal/VinUniversity/Wild-Diffusion/toy/trainer.py#L143)
- PDF draft tham chiếu mục tiêu theory:
  - [draft_my_method.md](/Users/quan238/personal/VinUniversity/Wild-Diffusion/pdfs/draft_my_method.md)
- Experiment outputs (server):
  - `/home/quanth/working_space/toy_outputs/baseline_attack_clean_anchor_warmup_diag_2ppm_logn_3k`
  - `/home/quanth/working_space/toy_outputs/baseline_attack_clean_anchor_diag_2ppm_logn_3k`
  - `/home/quanth/working_space/toy_outputs/baseline_attack_clean_anchor_warmup_diag_fixzero_2ppm_logn_3k`
  - `/home/quanth/working_space/toy_outputs/theory_check_lam0.2_seed0_2ppm_3k` ... `theory_check_lam0.4_seed2_2ppm_3k`

## 10) Repro Command (job quan trọng)
- Diagnostic warmup trước fix: job `6254`.
- Diagnostic no-warmup: job `6255`.
- Warmup sau fix zero-init: job `6259`.
- Theory sweep 9-run: job `6260`.

