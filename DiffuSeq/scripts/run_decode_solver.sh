CUDA_VISIBLE_DEVICES=7 python3.8 -u /home/kara/DiffuSeq/scripts/run_decode_solver.sh \
--model_dir /home/kara/DiffuSeq/diffusion_models/diffuseq_detox_h128_lr1e-05_t2000_sqrt_lossaware_seed102_learned_mask_fp16_denoise_0.5_reproduce20240608-05:30:31/ema_0.9999_050000.pt \
--seed 110 \
--bsz 100 \
--step 10 \
--split test
