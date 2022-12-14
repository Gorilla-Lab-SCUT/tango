python eval.py  \
--obj_path data/source_meshes/person.obj \
--output_dir results/demo/people/iron \
--prompt "a 3D rendering of the Iron Man in unreal engine" \
--if_normal_clamp \
--width 468 \
--background 'black' \
--init_r_and_s \
--init_roughness 0.7 \
--local_percentage 0.4 \
--symmetry \
--radius 2.0 \
--n_views 3 \
--material_random_pe_sigma  30 \
--material_random_pe_numfreq  256 \
--num_lgt_sgs 64 \
--n_normaugs 4 \
--n_augs 1 \
--frontview_std 2 \
--clipavg view \
--lr_decay 0.7 \
--mincrop 0.05 \
--maxcrop 0.2 \
--seed 150 \
--n_iter 1501 \
--learning_rate 0.0005   \
--model_dir results/demo/people/iron/iter1500.pth \
--render_gif
