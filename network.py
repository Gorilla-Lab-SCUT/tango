import torch
import torch.nn as nn
import numpy as np
from embedder import get_embedder,FourierFeatureTransform

### uniformly distribute points on a sphere
def fibonacci_sphere(samples=1):
    '''
    https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    :param samples:
    :return:
    '''
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])
    points = np.array(points)
    return points

def compute_energy(lgtSGs):
    lgtLambda = torch.abs(lgtSGs[:, 3:4])       # [M, 1]
    lgtMu = torch.abs(lgtSGs[:, 4:])               # [M, 3]
    energy = lgtMu * 2.0 * np.pi / lgtLambda * (1.0 - torch.exp(-2.0 * lgtLambda))
    return energy

class svbrdf_network(nn.Module):
    def __init__(self, 
                 material_random_pe_numfreq=128, 
                 dim=256,
                 material_random_pe_sigma=1,
                 white_specular=False,
                 white_light=False,
                 num_lgt_sgs=32,
                 num_base_materials=1,
                 upper_hemi=False,
                 fix_specular_albedo=False,
                 init_r_and_s = False,
                 init_roughness=0.7,
                 init_specular=0.23,
                 material_nerf_pe_numfreq=0):
        """_summary_

        Args:
            material_random_pe_numfreq (int, optional): the numer of frequencies using nerf's position encoding in svbrdf network. Defaults to 128.
            dim (int, optional): Dimension of the MLP layer. Defaults to 256.
            material_random_pe_sigma (int, optional): the sigma of random position encoding in svbrdf network. Defaults to 1.
            white_specular (bool, optional): _description_. Defaults to False.
            white_light (bool, optional): _description_. Defaults to False.
            num_lgt_sgs (int, optional): the number of light SGs. Defaults to 32.
            num_base_materials (int, optional): number of BRDF SG. Defaults to 1.
            upper_hemi (bool, optional): check if lobes are in upper hemisphere. Defaults to False.
            fix_specular_albedo (bool, optional): _description_. Defaults to False.
            init_r_and_s (bool, optional): It will initialize roughness and specular if setting true. Defaults to False.
            init_roughness (float, optional): Initial value of roughness 0~1. Defaults to 0.7.
            init_specular (float, optional): Initial value of specular 0~1. Defaults to 0.23.
            material_nerf_pe_numfreq (int, optional): the numer of frequencies using nerf's position encoding in svbrdf network. Defaults to 0.
        """
        super().__init__()
        init_roughness= torch.tensor(init_roughness)
        init_roughness= torch.arctanh(init_roughness*2-1)
        
        init_specular= torch.tensor(init_specular)
        init_specular= torch.arctanh(init_specular*2-1)
        self.activate = nn.PReLU()
        self.material_nerf_pe_numfreq=material_nerf_pe_numfreq
        base_layers=[]      
        if material_random_pe_numfreq >0:
            base_layers.append(FourierFeatureTransform(3, material_random_pe_numfreq, material_random_pe_sigma))
            base_layers.append(nn.Sequential(nn.Linear(material_random_pe_numfreq*2+3, dim),self.activate))
        elif material_nerf_pe_numfreq>0:       
            self.embed_fn, input_dim = get_embedder(material_nerf_pe_numfreq)
            base_layers.append(nn.Sequential(nn.Linear(input_dim, 256),self.activate))
          
        for i in range(1):
            base_layers.append(nn.Sequential(nn.Linear(dim, dim),self.activate))
        self.mlp_base = nn.ModuleList(base_layers)
        
        roughness_layers=[]
        for i in range(2):
            roughness_layers.append(nn.Sequential(nn.Linear(dim, dim),self.activate))
    
        a=nn.Linear(dim, 1)
        if init_r_and_s:
            torch.nn.init.constant_(a.weight, 0)
            torch.nn.init.constant_(a.bias, init_roughness)
        roughness_layers.append(a)
        roughness_layers.append(nn.Tanh())
        self.mlp_roughness = nn.ModuleList(roughness_layers)

        specular_layers=[]
        for i in range(2):
            specular_layers.append(nn.Sequential(nn.Linear(dim, dim),self.activate))
        b=nn.Linear(dim, 3)
        if init_r_and_s:
            torch.nn.init.constant_(b.weight, 0)
            torch.nn.init.constant_(b.bias, init_specular)
        specular_layers.append(b)
        specular_layers.append(nn.Tanh())
        self.mlp_specular = nn.ModuleList(specular_layers)
 
        diffuse_layers=[]
        for i in range(2):
            diffuse_layers.append(nn.Sequential(nn.Linear(dim, dim),self.activate))
        diffuse_layers.append(nn.Linear(dim, 3))
        diffuse_layers.append(nn.Tanh())
        self.mlp_diffuse = nn.ModuleList(diffuse_layers)
        
        
        self.numLgtSGs = num_lgt_sgs
        self.numBrdfSGs = num_base_materials
        print('Number of Light SG: ', self.numLgtSGs)
        print('Number of BRDF SG: ', self.numBrdfSGs)
        # by using normal distribution, the lobes are uniformly distributed on a sphere at initialization
        self.white_light = white_light
        if self.white_light:
            print('Using white light!')
            self.lgtSGs = nn.Parameter(torch.randn(self.numLgtSGs, 5), requires_grad=True)   # [M, 5]; lobe + lambda + mu
            # self.lgtSGs.data[:, -1] = torch.clamp(torch.abs(self.lgtSGs.data[:, -1]), max=0.01)
        else:
            self.lgtSGs = nn.Parameter(torch.randn(self.numLgtSGs, 7), requires_grad=True)   # [M, 7]; lobe + lambda + mu
            self.lgtSGs.data[:, -2:] = self.lgtSGs.data[:, -3:-2].expand((-1, 2))
            # self.lgtSGs.data[:, -3:] = torch.clamp(torch.abs(self.lgtSGs.data[:, -3:]), max=0.01)

        # make sure lambda is not too
        # close to zero
        self.lgtSGs.data[:, 3:4] = 20. + torch.abs(self.lgtSGs.data[:, 3:4] * 100.)
        # make sure total energy is around 1.
        energy = compute_energy(self.lgtSGs.data)
        # print('init envmap energy: ', torch.sum(energy, dim=0).clone().cpu().numpy())
        self.lgtSGs.data[:, 4:] = torch.abs(self.lgtSGs.data[:, 4:]) / torch.sum(energy, dim=0, keepdim=True) * 2. * np.pi
        energy = compute_energy(self.lgtSGs.data)
        print('init envmap energy: ', torch.sum(energy, dim=0).clone().cpu().numpy())

        # deterministicly initialize lobes
        lobes = fibonacci_sphere(self.numLgtSGs).astype(np.float32)
        self.lgtSGs.data[:, :3] = torch.from_numpy(lobes)
        # check if lobes are in upper hemisphere
        self.upper_hemi = upper_hemi
        if self.upper_hemi:
            print('Restricting lobes to upper hemisphere!')
            self.restrict_lobes_upper = lambda lgtSGs: torch.cat((lgtSGs[..., :1], torch.abs(lgtSGs[..., 1:2]), lgtSGs[..., 2:]), dim=-1)

            # limit lobes to upper hemisphere
            self.lgtSGs.data = self.restrict_lobes_upper(self.lgtSGs.data)

        self.white_specular = white_specular
        self.fix_specular_albedo = fix_specular_albedo
    
    def get_light(self):
        lgtSGs = self.lgtSGs.clone().detach()
        if self.white_light:
            lgtSGs = torch.cat((lgtSGs, lgtSGs[..., -1:], lgtSGs[..., -1:]), dim=-1)
        if self.upper_hemi:
            # limit lobes to upper hemisphere
            lgtSGs = self.restrict_lobes_upper(lgtSGs)

        return lgtSGs

    def forward(self, points):
        x=points
        if self.material_nerf_pe_numfreq>0:
            x= self.embed_fn(x)
        for layer in self.mlp_base:
            x = layer(x)
        
        roughness = self.mlp_roughness[0](x)
        for layer in self.mlp_roughness[1:]:
            roughness = layer(roughness)
        roughness = (roughness+1)/2
        
        specular_reflectacne = self.mlp_specular[0](x)
        for layer in self.mlp_specular[1:]:
            specular_reflectacne = layer(specular_reflectacne)
        specular_reflectacne = (specular_reflectacne+1)/2
        
        diffuse_albedo = self.mlp_diffuse[0](x)
        for layer in self.mlp_diffuse[1:]:
            diffuse_albedo = layer(diffuse_albedo)
        diffuse_albedo = (diffuse_albedo+1)/2
        
        lgtSGs = self.lgtSGs
        if self.white_light:
            lgtSGs = torch.cat((lgtSGs, lgtSGs[..., -1:], lgtSGs[..., -1:]), dim=-1)
        if self.upper_hemi:
            # limit lobes to upper hemisphere
            lgtSGs = self.restrict_lobes_upper(lgtSGs)
        # roughness[:,:] = 0
        # specular_reflectacne[:,:]= 0.4
        ret = dict([
            ('sg_lgtSGs', lgtSGs),
            ('sg_specular_reflectance', specular_reflectacne),
            ('sg_roughness', roughness),
            ('sg_diffuse_albedo', diffuse_albedo)
        ])
        return ret
    
    #Convert rectangular coordinate system to spherical coordinate system
def cart2sph(cart):#cart:[batch,3] 3 include x,y,z
    hxy = torch.hypot(cart[:,0], cart[:,1])
    r = torch.hypot(hxy, cart[:,2]).view(-1,1)
    theta = torch.atan2(hxy,cart[:,2] ).view(-1,1)
    phi = torch.atan2 (cart[:,1] , cart[:,0]).view(-1,1)
    sph = torch.cat((r,theta,phi),dim=1)
    return sph

    #Convert spherical coordinate system to rectangular coordinate system
def sph2cart(sph):
    rsin_theta = sph[:,0] * torch.sin(sph[:,1])
    x = (rsin_theta * torch.cos(sph[:,2])).view(-1,1)
    y = (rsin_theta * torch.sin(sph[:,2])).view(-1,1)
    z = (sph[:,0] * torch.cos(sph[:,1])).view(-1,1)
    cart = torch.cat((x,y,z),dim=1)
    return cart

class Normal_estimation_network(nn.Module):
    def __init__(self,
                 normal_nerf_pe_numfreq=True,
                 normal_random_pe_numfreq=False,
                 max_delta_theta=np.pi/3,
                 max_delta_phi=np.pi/3,
                 normal_random_pe_sigma=20,
                 if_normal_clamp = False):
        super().__init__()
        self.normal_nerf_pe_numfreq = normal_nerf_pe_numfreq
        self.normal_random_pe_numfreq = normal_random_pe_numfreq
        self.max_delta_theta=max_delta_theta
        self.max_delta_phi=max_delta_phi
        self.if_normal_clamp = if_normal_clamp
        self.activate = nn.PReLU()
        normal_layers=[]
        if self.normal_nerf_pe_numfreq>0:
            self.embed_fn, input_dim = get_embedder(self.normal_nerf_pe_numfreq)
            normal_layers.append(nn.Sequential(nn.Linear( input_dim+2 , 256),self.activate))
        elif self.normal_random_pe_numfreq>0:
            normal_layers.append(FourierFeatureTransform(3, self.normal_random_pe_numfreq, normal_random_pe_sigma))
            normal_layers.append(nn.Sequential(nn.Linear(self.normal_random_pe_numfreq*2+3+2, 256),self.activate))
        else:
            normal_layers.append(nn.Sequential(nn.Linear(5, 256),self.activate))

        for i in range(2):
            normal_layers.append(nn.Sequential(nn.Linear(256, 256),self.activate))
        normal_layers.append(nn.Linear(256, 2))
        normal_layers.append(nn.Tanh())
        self.mlp_normal = nn.ModuleList(normal_layers)
    def forward(self, points, normals):
        normals_sph = cart2sph(normals)
        normals_sph2 = normals_sph[:,1:]
        x = points
        if self.normal_nerf_pe_numfreq>0:
            x = self.embed_fn(x)
            x = torch.cat((x,normals_sph2),dim=1)
            for layer in self.mlp_normal:
                x = layer(x)
        elif self.normal_random_pe_numfreq>0:
            x = self.mlp_normal[0](x)
            x = torch.cat((x,normals_sph2),dim=1)
            for layer in self.mlp_normal[1:]:
                x = layer(x)
        else:
            x = torch.cat((x,normals_sph2),dim=1)
            for layer in self.mlp_normal:
                x = layer(x)
        
        x = x*torch.tensor([self.max_delta_theta,self.max_delta_phi]).cuda()
        normals_sph[:,1]+=x[:,0]
        normals_sph[:,2]+=x[:,1]
        # ipdb.set_trace()
        if self.if_normal_clamp:
            normals_sph[:,1] = torch.clamp(normals_sph[:,1].clone(), 0, np.pi)
            normals_sph[:,2] = torch.clamp(normals_sph[:,2].clone(), 0, np.pi*2)
        normals_disp = sph2cart(normals_sph)
        return normals_disp
        
