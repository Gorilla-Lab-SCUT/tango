python eval.py  \
--obj_path data/source_meshes/shoe.obj \
--output_dir results/demo/shoe/gold1111 \
--prompt "a shoe made of gold" \
--init_r_and_s \
--init_roughness 0.3 \
--max_delta_theta 0 \
--max_delta_phi 0 \
--width 512 \
--local_percentage 0.7 \
--background 'gaussian' \
--radius 1.5 \
--n_views 1 \
--material_random_pe_sigma 0.5 \
--material_random_pe_numfreq  3 \
--num_lgt_sgs 64 \
--n_normaugs 4 \
--n_augs 1 \
--frontview_std 8 \
--clipavg view \
--lr_decay 0.7 \
--mincrop 0.1 \
--maxcrop 0.1 \
--seed 150 \
--n_iter 501  \
--learning_rate 0.0005  \
--frontview_center 0.5 0.6283 \
--model_dir results/demo/shoe/gold1111/iter500.pth \
--render_gif