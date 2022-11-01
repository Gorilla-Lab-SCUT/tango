import clip
from tqdm import tqdm
import torch
from neural_style_field import NeuralStyleField
from utils import device
from utils import clip_model
import numpy as np
import random
import torchvision
import os
import argparse
from pathlib import Path
from torchvision import transforms
import open3d as o3d

def get_normalize_mesh(pro_path):
    mesh = o3d.io.read_triangle_mesh(pro_path)
    vertices = np.asarray(mesh.vertices)
    shift = np.mean(vertices,axis=0)
    scale = np.max(np.linalg.norm(vertices-shift, ord=2, axis=1))
    vertices = (vertices-shift) / scale
    mesh.vertices = o3d.cuda.pybind.utility.Vector3dVector(vertices)
    return mesh
    
def train(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float32)
    # torch.set_num_threads(8)
    # Constrain all sources of randomness
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    objbase, extension = os.path.splitext(os.path.basename(args.obj_path))
    # Check that isn't already done
    if (not args.overwrite) and os.path.exists(os.path.join(args.output_dir, "loss.png")) and \
            os.path.exists(os.path.join(args.output_dir, f"{objbase}_final.obj")):
        print(f"Already done with {args.output_dir}")
        exit()
    elif args.overwrite and os.path.exists(os.path.join(args.output_dir, "loss.png")) and \
            os.path.exists(os.path.join(args.output_dir, f"{objbase}_final.obj")):
        import shutil
        for filename in os.listdir(args.output_dir):
            file_path = os.path.join(args.output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    losses = []
    n_augs = args.n_augs
    dir = args.output_dir
    # global transformation
    augment_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(1, 1)), #Obtain a thumbnail image to meet the requirements of clip's input image size
    ])
    # local transformation
    normaugment_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(args.mincrop, args.maxcrop)),
    ])
    normweight = 1.0
    model = NeuralStyleField(args.material_random_pe_numfreq,
                             args.material_random_pe_sigma,
                             args.num_lgt_sgs,
                             args.max_delta_theta,
                             args.max_delta_phi,
                             args.normal_nerf_pe_numfreq,
                             args.normal_random_pe_numfreq,
                             args.symmetry,
                             args.radius,
                             args.background,
                             args.init_r_and_s,
                             args.width,
                             args.init_roughness,
                             args.init_specular,
                             args.material_nerf_pe_numfreq,
                             args.normal_random_pe_sigma,
                             args.if_normal_clamp
                            )
    if torch.cuda.is_available():
        model.cuda()

     
    model.train()
    optim = torch.optim.AdamW(model.parameters(), args.learning_rate)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,
                                                        [500,1000],
                                                        args.lr_decay)
   
    if args.prompt:
        prompt = ' '.join(args.prompt)
        prompt_token = clip.tokenize([prompt]).to(device)
        encoded_text = clip_model.encode_text(prompt_token)

        # Save prompt
        with open(os.path.join(dir, prompt), "w") as f:
            f.write("")

        norm_encoded = encoded_text
    # ipdb.set_trace()

    mesh = get_normalize_mesh(args.obj_path)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    trainer = tqdm(range(args.n_iter))
    for i in trainer:
        optim.zero_grad()

        
        
        rendered_images = model(scene=scene, 
                                num_views=args.n_views,
                                center_azim=args.frontview_center[0],
                                center_elev=args.frontview_center[1],
                                std=args.frontview_std,
                                )
        rendered_images = rendered_images.cuda()
   

        if n_augs > 0:
            loss = 0.0
            for _ in range(n_augs):
                augmented_image = augment_transform(rendered_images[:,0:3,:,:])
                if i % 20 == 0:
                    torchvision.utils.save_image(augmented_image, os.path.join(dir, 'iter_global{}.jpg'.format(i)))
                encoded_renders = clip_model.encode_image(augmented_image)
                if args.prompt: 
                    if args.clipavg == "view":
                        if encoded_text.shape[0] > 1:
                            loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                            torch.mean(encoded_text, dim=0), dim=0)
                        else:
                            loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                            encoded_text)
                    else:
                        loss -= torch.mean(torch.cosine_similarity(encoded_renders, encoded_text))
                    
        loss.backward(retain_graph=True)
        if args.n_normaugs > 0:
            normloss = 0.0
            for _ in range(args.n_normaugs):
                augmented_image = normaugment_transform(rendered_images)
                shape = augmented_image.shape[0]*augmented_image.shape[2]*augmented_image.shape[3]
                object_percent = torch.sum(augmented_image[:,3,:,:]==1) / shape
                while object_percent <= args.local_percentage: 
                    augmented_image = normaugment_transform(rendered_images)
                    object_percent = torch.sum(augmented_image[:,3,:,:]==1) / shape

                augmented_image = augmented_image[:,0:3,:,:]
                if i % 20 == 0:
                    torchvision.utils.save_image(augmented_image, os.path.join(dir, 'iter_local{}.jpg'.format(i)))
                encoded_renders = clip_model.encode_image(augmented_image)
                if args.prompt:
                    if args.clipavg == "view":
                        if norm_encoded.shape[0] > 1:
                            normloss -= normweight * torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                                                torch.mean(norm_encoded, dim=0),
                                                                                dim=0)
                        else:
                            normloss -= normweight * torch.cosine_similarity(
                                torch.mean(encoded_renders, dim=0, keepdim=True),
                                norm_encoded)
                    else:
                        normloss -= normweight * torch.mean(
                            torch.cosine_similarity(encoded_renders, norm_encoded))
            normloss.backward(retain_graph=True)
        optim.step()
        lr_scheduler.step()
        with torch.no_grad():
            losses.append(loss.item())
        if args.decayfreq is not None:
            if i % args.decayfreq == 0:
                normweight *= args.cropdecay
        if i % 100 == 0:
            # report_process(args, dir, i, loss, loss_check, losses, rendered_images[:,0:3,:,:])
            print('iter: {} loss: {}'.format(i, np.mean(losses[-100:])))
            torchvision.utils.save_image(rendered_images[:,0:3,:,:], os.path.join(dir, 'iter_{}.jpg'.format(i)))
            torch.save({'model': model.state_dict()}, os.path.join(dir, f'iter{i:03d}.pth'))
        lr = optim.state_dict()['param_groups'][0]['lr']
        trainer.set_description(desc=f'lr:{lr}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_lgt_sgs', type=int, default=32) #the number of light SGs
    parser.add_argument('--max_delta_theta', type=float, default=1.5707) #maximum offset of elevation angle whose unit is radian
    parser.add_argument('--max_delta_phi', type=float, default=1.5707) #maximum offset of azimuth angle whose unit is radian
    
    parser.add_argument('--normal_nerf_pe_numfreq',  type=int, default=0) #the number of frequencies using nerf's position encoding in normal network
    parser.add_argument('--normal_random_pe_numfreq', type=int, default=0) #the number of frequencies using random position encoding in normal network
    parser.add_argument('--normal_random_pe_sigma', type=float, default=20.0) #the sigma of random position encoding in normal network
    parser.add_argument('--material_nerf_pe_numfreq',  type=int, default=0) #the numer of frequencies using nerf's position encoding in svbrdf network
    parser.add_argument('--material_random_pe_numfreq', type=int, default=0) #the numer of frequencies using random position encoding in svbrdf network
    parser.add_argument('--material_random_pe_sigma', type=float, default=20.0) #the sigma of random position encoding in svbrdf network
    parser.add_argument('--if_normal_clamp', action='store_true') 
    
    parser.add_argument('--init_r_and_s', action='store_true') #It will initialize roughness and specular if setting true
    parser.add_argument('--init_roughness', type=float, default=0.7) #Initial value of roughness 0~1
    parser.add_argument('--init_specular', type=float, default=0.23)  #Initial value of specular 0~1
    parser.add_argument('--width', type=int, default=512) #the size of render image will be [width,width]
    
    parser.add_argument('--radius', type=float, default=2.0) #the sampling raidus of camara position
    parser.add_argument('--background', type=str, default='black') #the background of render image.'black','white' or 'gaussian' can be selected
    parser.add_argument('--local_percentage',type=float, default=0.7) #percent threshold of the object's mask in cropped image.It will be cropped again
                                                                      #if the proportion of the object's mask in cropped image is less than this threshold.
                                                                      #This parameter can effectively prevent image degradation
    parser.add_argument('--obj_path', type=str, default='meshes/mesh1.obj') #the storage path of raw or original mesh
    parser.add_argument('--prompt', nargs="+", default='a pig with pants') #the text prompt to style a raw mesh
    parser.add_argument('--output_dir', type=str, default='round2/alpha5') #directory where the results will be saved
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--lr_decay', type=float, default=1) #decay factor of learning rate
    parser.add_argument('--n_views', type=int, default=4) #number of viewpoints optimized at the same time in an iteration
    parser.add_argument('--n_augs', type=int, default=0) #In one iteration, the gradient retrieval times of the rendered thumbnail
    parser.add_argument('--n_normaugs', type=int, default=0) #In one iteration, the gradient retrieval times of the local clip of the rendered image
    parser.add_argument('--n_iter', type=int, default=1501) #number of iterations

    parser.add_argument('--frontview_std', type=float, default=8) # Angular variance of the off-center view
    parser.add_argument('--frontview_center', nargs=2, type=float, default=[0., 0.]) #Center position of viewpoint.[azimuth angle(0~2π),elevation angle(0~π)]
    parser.add_argument('--clipavg', type=str, default=None)
    parser.add_argument('--mincrop', type=float, default=1) #minimium clipping scale in 2D augmentation 
    parser.add_argument('--maxcrop', type=float, default=1) #maximium clipping scale in 2D augmentation
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--seed', type=int, default=0) #random seed
    parser.add_argument('--symmetry', default=False, action='store_true') #With this symmetry prior, the texture of the mesh 
                                                                          #will be symmetrical along the z-axis.We use this parameter in person
    parser.add_argument('--decayfreq', type=int, default=None) #decay freaquency of learning rate
    args = parser.parse_args()
 
    train(args)

   
