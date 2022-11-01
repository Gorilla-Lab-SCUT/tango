import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch
from utils import device
from sg_render import render_with_sg
from network import svbrdf_network
from network import Normal_estimation_network
import open3d as o3d
import numpy as np
import ipdb

width1 = 712
def get_rays(elev, azim, r=3.0,width = 512):
    x = r * torch.cos(elev) * torch.cos(azim)
    y = r * torch.sin(elev)
    z = r * torch.cos(elev) * torch.sin(azim)
    # print(elev,azim,x,y,z)

    pos = np.array([x.numpy(),y.numpy(),z.numpy()])
    look_at = np.array([-x.numpy(),-y.numpy(),-z.numpy()])
    # direction = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0)

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(fov_deg=60,
                                                              center=look_at,
                                                              eye=pos,
                                                              up=[0, -1, 0],
                                                              width_px=width,
                                                              height_px=width,
                                                             )
    
    return rays

def get_rays1(elev, azim, r=3.0):
    x = r * torch.cos(elev) * torch.cos(azim)
    y = r * torch.sin(elev)
    z = r * torch.cos(elev) * torch.sin(azim)
    # print(elev,azim,x,y,z)

    pos = np.array([x.numpy(),y.numpy(),z.numpy()])
    look_at = np.array([-x.numpy(),-y.numpy(),-z.numpy()])
    # direction = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0)

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(fov_deg=60,
                                                              center=look_at,
                                                              eye=pos,
                                                              up=[0, -1, 0],
                                                              width_px=width1,
                                                              height_px=width1,
                                                             )
    
    return rays

class NeuralStyleField(nn.Module):
    def __init__(self,  
                 material_random_pe_numfreq=0,
                 material_random_pe_sigma=12,
                 num_lgt_sgs=32,
                 max_delta_theta=np.pi/2,
                 max_delta_phi=np.pi/2,
                 normal_nerf_pe_numfreq=0,
                 normal_random_pe_numfreq=0,
                 symmetry=False,
                 radius=2.0,
                 background='black',
                 init_r_and_s=False,
                 width=512,
                 init_roughness=0.7,
                 init_specular=0.23,
                 material_nerf_pe_numfreq=0,
                 normal_random_pe_sigma=20,
                 if_normal_clamp=False):
        """_summary_

        Args:
            material_random_pe_numfreq (int, optional): the numer of frequencies using nerf's position encoding in svbrdf network. Defaults to 0.
            material_random_pe_sigma (int, optional): the sigma of random position encoding in svbrdf network. Defaults to 12.
            num_lgt_sgs (int, optional): the number of light SGs. Defaults to 32.
            max_delta_theta (_type_, optional): maximum offset of elevation angle whose unit is radian. Defaults to np.pi/2.
            max_delta_phi (_type_, optional): maximum offset of azimuth angle whose unit is radian. Defaults to np.pi/2.
            normal_nerf_pe_numfreq (int, optional): the number of frequencies using nerf's position encoding in normal network. Defaults to 0.
            normal_random_pe_numfreq (int, optional): the number of frequencies using random position encoding in normal network. Defaults to 0.
            symmetry (bool, optional): With this symmetry prior, the texture of the mesh will be symmetrical along the z-axis.We use this parameter in person. Defaults to False.
            radius (float, optional): the sampling raidus of camara position. Defaults to 2.0.
            background (str, optional): the background of render image.'black','white' or 'gaussian' can be selected. Defaults to 'black'.
            init_r_and_s (bool, optional): It will initialize roughness and specular if setting true. Defaults to False.
            width (int, optional): the size of render image will be [width,width]. Defaults to 512.
            init_roughness (float, optional): Initial value of roughness 0~1. Defaults to 0.7.
            init_specular (float, optional): Initial value of specular 0~1. Defaults to 0.23.
            material_nerf_pe_numfreq (int, optional): the numer of frequencies using nerf's position encoding in svbrdf network. Defaults to 0.
            normal_random_pe_sigma (int, optional): the sigma of random position encoding in normal network. Defaults to 20.
            if_normal_clamp (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        self.svbrdf_network = svbrdf_network(material_random_pe_numfreq = material_random_pe_numfreq,
                                                              material_random_pe_sigma = material_random_pe_sigma,
                                                            dim=256,
                                                            white_specular = False,
                                                            white_light = True,
                                                            num_lgt_sgs = num_lgt_sgs,
                                                            num_base_materials = 1,
                                                            upper_hemi = False,
                                                            fix_specular_albedo = False,                                      
                                                            init_r_and_s = init_r_and_s,
                                                            init_roughness=init_roughness,
                                                            init_specular=init_specular,
                                                            material_nerf_pe_numfreq= material_nerf_pe_numfreq)
        self.radius = radius
        self.symmetry = symmetry
        self.width = width
        self.elev = 0.6283
        self.azim = 0.5
        self.Normal_estimation_network=Normal_estimation_network(max_delta_theta=max_delta_theta,
                                                                 max_delta_phi=max_delta_phi,
                                                                 normal_nerf_pe_numfreq=normal_nerf_pe_numfreq,
                                                                 normal_random_pe_numfreq=normal_random_pe_numfreq,
                                                                 normal_random_pe_sigma=normal_random_pe_sigma,
                                                                 if_normal_clamp = if_normal_clamp)
        if background=='black':
            self.background =  torch.zeros(width*width,3)
        if background=='white':
            self.background =  torch.ones(width*width,3)
        if background=='gaussian':
            self.background =  torch.randn(width*width,3)
            self.background[:,1] =self.background[:,0]
            self.background[:,2] =self.background[:,0]

    def render_single_image(self, scene, azim , elev):
        
        images = []
        normal1 = []
        normal2 = []

        roughness = []
        specular = []
        diffuse = []

        rays = get_rays1(elev, azim, r=2)
        ans = scene.cast_rays(rays)
        
        hit = ans['t_hit'].isfinite()
        points = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1))
        normal = ans['primitive_normals'][hit].reshape((-1,3))
        view_dirs = -torch.nn.functional.normalize(torch.from_numpy(rays[hit][:,3:].numpy())).to(device)
        pcd = o3d.t.geometry.PointCloud(points)
        pcd.point["normals"] = normal 
        pcd = pcd.to_legacy()
        points = torch.from_numpy(np.asarray(pcd.points)).float().to(device)
        if self.symmetry:
            points[:,2]=torch.abs(points[:,2])
        normals1 = torch.nn.functional.normalize(torch.from_numpy(np.asarray(pcd.normals))).float().to(device)

        normals2 = self.Normal_estimation_network(points,normals1)
        normals2 = torch.nn.functional.normalize(normals2)

        ret = self.get_rbg_value(points,normals2, view_dirs)
        hit1 = torch.from_numpy(hit.reshape(width1*width1).numpy())
        sg_rgb_values = torch.ones(width1*width1,3).float().to(device)
        sg_rgb_values[hit1] = ret['sg_rgb']
        sg_normal1_values = torch.ones(width1*width1,3).float().to(device)
        sg_normal1_values[hit1]= normals1
        sg_normal1_values = sg_normal1_values.reshape(1,width1,width1,3)
        sg_normal2_values = torch.ones(width1*width1,3).float().to(device)
        sg_normal2_values[hit1]= normals2
        sg_normal2_values = sg_normal2_values.reshape(1,width1,width1,3)

        sg_roughness_values = torch.ones(width1*width1,1).float().to(device)
        sg_roughness_values[hit1] = ret['sg_roughness']
        sg_roughness_values = sg_roughness_values.reshape(1,width1,width1,1)
        sg_roughness_values = torch.clamp(sg_roughness_values, 0, 1)

        sg_diffuse_values = torch.ones(width1*width1,3).float().to(device)
        sg_diffuse_values[hit1] = ret['sg_diffuse_rgb']
        sg_diffuse_values = sg_diffuse_values.reshape(1,width1,width1,3)
        sg_diffuse_values = torch.clamp(sg_diffuse_values, 0, 1)

        sg_specular_values = torch.ones(width1*width1,3).float().to(device)
        sg_specular_values[hit1] = ret['sg_specular_rgb']
        sg_specular_values = sg_specular_values.reshape(1,width1,width1,3)
        sg_specular_values = torch.clamp(sg_specular_values, 0, 1)


        normal1.append(sg_normal1_values)
        normal2.append(sg_normal2_values)

        roughness.append(sg_roughness_values)
        diffuse.append(sg_diffuse_values)
        specular.append(sg_specular_values)
        
        image = sg_rgb_values.reshape(width1,width1,3).unsqueeze(0)
        
        image = torch.clamp(image, 0, 1)
        images.append(image)
        images = torch.cat(images, dim=0).permute(0, 3, 1, 2)

        normal1 = torch.cat(normal1, dim=0).permute(0, 3, 1, 2)
        normal2 = torch.cat(normal2, dim=0).permute(0, 3, 1, 2)

        roughness = torch.cat(roughness, dim=0).permute(0, 3, 1, 2)
        diffuse = torch.cat(diffuse, dim=0).permute(0, 3, 1, 2)
        specular = torch.cat(specular, dim=0).permute(0, 3, 1, 2)
        return images, normal1, normal2, roughness, diffuse, specular 

    def forward(self, scene, num_views=8, std=8, center_elev=0, center_azim=0):
        if num_views>1:
            self.elev = torch.cat((torch.tensor([center_elev]), torch.randn(num_views - 1) * np.pi / std + center_elev))
            self.azim = torch.cat((torch.tensor([center_azim]),torch.randn(num_views - 1) * 2 * np.pi / std + center_azim))
        if num_views==1:
            self.elev =  torch.randn(num_views) * np.pi /std+ center_elev
            self.azim += torch.rand(num_views) * 0.1
        images_and_masks = []
        for i in range(num_views):
            rays = get_rays(self.elev[i], self.azim[i], r=self.radius,width=self.width)
            ans = scene.cast_rays(rays)
            
            hit = ans['t_hit'].isfinite()
            points = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1))
            normal = ans['primitive_normals'][hit].reshape((-1,3))
            # import ipdb
            # ipdb.set_trace()
            view_dirs = -torch.nn.functional.normalize(torch.from_numpy(rays[hit][:,3:].numpy())).to(device)
            pcd = o3d.t.geometry.PointCloud(points)
            pcd.point["normals"] = normal 
            pcd = pcd.to_legacy()
            points = torch.from_numpy(np.asarray(pcd.points)).float().to(device)
            if self.symmetry:
                points[:,2]=torch.abs(points[:,2])
            normals1= torch.nn.functional.normalize(torch.from_numpy(np.asarray(pcd.normals))).float().to(device)

            normals2 = self.Normal_estimation_network(points,normals1)
            normals2 = torch.nn.functional.normalize(normals2)

            ret = self.get_rbg_value(points, normals2, view_dirs) 
            hit1 = torch.from_numpy(hit.reshape(self.width*self.width).numpy())
            sg_rgb_values = self.background.float().to(device)
            sg_rgb_values[hit1] = ret['sg_rgb']

            mask = torch.from_numpy(hit.numpy()).float().cuda().reshape(1,self.width,self.width,1)
            image = sg_rgb_values.reshape(self.width,self.width,3).unsqueeze(0)   
            image = torch.clamp(image, 0, 1)
            image_and_mask = torch.cat((image,mask),dim=3)
            images_and_masks.append(image_and_mask)
        images_and_masks = torch.cat(images_and_masks, dim=0).permute(0, 3, 1, 2)
        
        return images_and_masks
    
    def get_rbg_value(self, points,normals, view_dirs):
        ret = { }
        sg_envmap_material = self.svbrdf_network(points)
        sg_ret = render_with_sg(lgtSGs=sg_envmap_material['sg_lgtSGs'],
                                specular_reflectance=sg_envmap_material['sg_specular_reflectance'],
                                roughness=sg_envmap_material['sg_roughness'],
                                diffuse_albedo=sg_envmap_material['sg_diffuse_albedo'],
                                normal=normals, viewdirs=view_dirs,
                               )
        ret.update(sg_ret)
        return ret




