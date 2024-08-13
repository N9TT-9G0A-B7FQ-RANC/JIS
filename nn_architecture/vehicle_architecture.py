import torch
import numpy as np
import json
  
class SimplifiedPacejka(torch.nn.Module):

    def __init__(self, parameters):
         
        super(SimplifiedPacejka, self).__init__()

        for key, value in parameters.items():
            if not value[3]:
                param_init = torch.nn.Parameter(torch.tensor(np.random.normal()).unsqueeze(dim=0).float())
                self.set_parameter(name = key, value = param_init, mean = value[0], std = value[1], requires_grad = value[2])
            else:
                param_init = torch.nn.Parameter(torch.tensor(0).unsqueeze(dim=0).float())
                self.set_parameter(name = key, value = param_init, mean = value[0], std = value[1], requires_grad = value[2])
    
    def set_parameter(self, name, value, mean, std, requires_grad):
        self.register_parameter(name=name, param=torch.nn.Parameter(torch.tensor(value).unsqueeze(dim=0).float(), requires_grad=requires_grad))
        self.register_parameter(name=f"{name}_mean", param=torch.nn.Parameter(torch.tensor(mean).unsqueeze(dim=0).float(), requires_grad=requires_grad))
        self.register_parameter(name=f"{name}_std", param=torch.nn.Parameter(torch.tensor(std).unsqueeze(dim=0).float(), requires_grad=requires_grad))

    def get_parameter(self, name):
        param = getattr(self, name)
        param_mean = getattr(self, f"{name}_mean")
        param_std = getattr(self, f"{name}_std")
        return param_mean + param * param_std
    
    def get_longitudinal_force(self, sigma):
        Bx = self.get_parameter('Bx')
        Cx = self.get_parameter('Cx')
        Dx = self.get_parameter('Dx')
        Ex = self.get_parameter('Ex')
        return Dx * torch.sin(Cx * torch.arctan(Bx * sigma - Ex * (Bx * sigma - torch.arctan(Bx * sigma))))
    
    def get_lateral_force(self, alpha):
        By = self.get_parameter('By')
        Cy = self.get_parameter('Cy')
        Dy = self.get_parameter('Dy')
        Ey = self.get_parameter('Ey')
        Sh = self.get_parameter('Sh')
        Sv = self.get_parameter('Sv')
        x = Sh + alpha
        return Dy * torch.sin(Cy * torch.arctan(By * x - Ey * (By * x - torch.arctan(By * x)))) + Sv
    
class VehicleBase(torch.nn.Module):

    def __init__(self):
        
        super(VehicleBase, self).__init__()

    def set_parameter(self, name, value, mean, std, requires_grad):
        self.register_parameter(name=name, param=torch.nn.Parameter(torch.tensor(value).unsqueeze(dim=0).float(), requires_grad=requires_grad))
        self.register_parameter(name=f"{name}_mean", param=torch.nn.Parameter(torch.tensor(mean).unsqueeze(dim=0).float(), requires_grad=requires_grad))
        self.register_parameter(name=f"{name}_std", param=torch.nn.Parameter(torch.tensor(std).unsqueeze(dim=0).float(), requires_grad=requires_grad))

    def set_parameters(self, names, values):
        for name, value in zip(names, values):  
            self.register_parameter(name=name, param=torch.nn.Parameter(torch.tensor([value]).unsqueeze(dim=0).float()))
        # for name, value in zip(names, values):  
        #     self.tire1.register_parameter(name=name, param=torch.nn.Parameter(torch.tensor(value).unsqueeze(dim=0).float()))
        # for name, value in zip(names, values):  
        #     self.tire2.register_parameter(name=name, param=torch.nn.Parameter(torch.tensor(value).unsqueeze(dim=0).float()))

    def get_parameter(self, name, standardized=False):
        param = getattr(self, name)
        if standardized:
            return param
        else:
            param_mean = getattr(self, f"{name}_mean")
            param_std = getattr(self, f"{name}_std")
            return param_mean + param * param_std

    def get_slipping_angle(self, vx, vy, delta):
        v = torch.clip(delta - torch.arctan(vy / vx), -1, 1)
        return v
    
    def get_slipping_rate(self, vxp, omega, r):
        d = torch.max(torch.concat((torch.abs(vxp), torch.abs(r * omega), torch.abs(r * omega - vxp)), dim=1),dim=1)[0].unsqueeze(1)
        return (r * omega - vxp) / d # torch.clip((r * omega - vxp) / d, -1, 1)
    
class duffing_oscillator(VehicleBase):

    def __init__(
            self,
            parameters_path,
            dt,
            output_format = 'acceleration',
            seed=42):
        
        self.output_format = output_format

        with open(parameters_path, 'r') as file: parameters = json.load(file)
        self.parameter = parameters
        super(duffing_oscillator, self).__init__()

        np.random.seed(seed)
        for key, value in parameters["global"].items():
            if not value[3]:
                param_init = torch.nn.Parameter(torch.tensor(np.random.normal()).unsqueeze(dim=0).float())
                self.set_parameter(name = key, value = param_init, mean = value[0], std = value[1], requires_grad = value[2])
            else:
                param_init = torch.nn.Parameter(torch.tensor(0).unsqueeze(dim=0).float())
                self.set_parameter(name = key, value = param_init, mean = value[0], std = value[1], requires_grad = value[2])

        self.state_variables = ['vy', 'psidt']
        self.register_buffer('dt', torch.tensor([dt]).float())
        self.register_buffer('z_scaling', torch.tensor(False))
        self.register_buffer('std', torch.tensor(torch.tensor([[1] * len(self.state_variables)]).float()))
        self.register_buffer('mean', torch.tensor([[0] * len(self.state_variables)]).float())

        self.state_variables = ['x', 'y']
        self.register_buffer('dt', torch.tensor([dt]).float())

    def forward(self, X, U):

        alpha = self.get_parameter('alpha')
        gamma = self.get_parameter('gamma')
        beta = self.get_parameter('beta')

        x0, x1  = torch.split(X, split_size_or_sections=1, dim=1)
        f = U
        d_x0__dt = x1
        d_x1__dt = -gamma * x1 - alpha * x0 - beta * x0**3 + f

        if self.output_format == 'acceleration':
            torch.concat((d_x0__dt, d_x1__dt), dim=1)
        else:
            return torch.concat((d_x0__dt, d_x1__dt), dim=1) * self.dt
        
    def sample(self):
        for key, value in self.parameter["global"].items():
            if not value[3]:
                param_init = torch.nn.Parameter(torch.tensor(np.random.normal()).unsqueeze(dim=0).float())
                self.set_parameter(name = key, value = param_init, mean = value[0], std = value[1], requires_grad = value[2])
            else:
                param_init = torch.nn.Parameter(torch.tensor(0).unsqueeze(dim=0).float())
                self.set_parameter(name = key, value = param_init, mean = value[0], std = value[1], requires_grad = value[2])
       
        
    def get_parameters(self, standardize = False):
        parameters = {}
        # value : mean, std, gradient, not random
        for key, value in self.parameter['global'].items():
            if value[2]:
                parameters[key] = self.get_parameter(key, standardize).item()
        return parameters
    
class TorchDOF2(VehicleBase):

    def __init__(
            self,
            parameters_path,
            dt,
            output_format = 'acceleration',
            seed=42,
        ):
        
        self.output_format = output_format

        with open(parameters_path, 'r') as file: parameters = json.load(file)
        self.parameter = parameters
        super(TorchDOF2, self).__init__()

        np.random.seed(seed)
        for key, value in parameters["global"].items():
            if not value[3]:
                param_init = torch.nn.Parameter(torch.tensor(np.random.normal()).unsqueeze(dim=0).float())
                self.set_parameter(name = key, value = param_init, mean = value[0], std = value[1], requires_grad = value[2])
            else:
                param_init = torch.nn.Parameter(torch.tensor(0).unsqueeze(dim=0).float())
                self.set_parameter(name = key, value = param_init, mean = value[0], std = value[1], requires_grad = value[2])

        self.state_variables = ['vy', 'psidt']
        self.register_buffer('dt', torch.tensor([dt]).float())
        self.register_buffer('z_scaling', torch.tensor(False))
        self.register_buffer('std', torch.tensor(torch.tensor([[1] * len(self.state_variables)]).float()))
        self.register_buffer('mean', torch.tensor([[0] * len(self.state_variables)]).float())

    def set_z_scale(self, z_scale):
        self.register_buffer('z_scaling', torch.tensor(z_scale))
    
    def set_std(self, std):
        self.register_buffer('std', torch.tensor([std]).float())
    
    def set_mean(self, mean):
        self.register_buffer('mean', torch.tensor([mean]).float())

    def sample(self):
        for key, value in self.parameter["global"].items():
            if not value[3]:
                param_init = torch.nn.Parameter(torch.tensor(np.random.normal()).unsqueeze(dim=0).float())
                self.set_parameter(name = key, value = param_init, mean = value[0], std = value[1], requires_grad = value[2])
            else:
                param_init = torch.nn.Parameter(torch.tensor(0).unsqueeze(dim=0).float())
                self.set_parameter(name = key, value = param_init, mean = value[0], std = value[1], requires_grad = value[2])
       
    def get_lateral_force(self, alpha, index):
        By = self.get_parameter(f'By{index}')
        Cy = self.get_parameter(f'Cy{index}')
        Dy = self.get_parameter(f'Dy{index}')
        Ey = self.get_parameter(f'Ey{index}')
        Sh = self.get_parameter(f'Sh{index}')
        Sv = self.get_parameter(f'Sv{index}')
        x = Sh + alpha
        return Dy * torch.sin(Cy * torch.arctan(By * x - Ey * (By * x - torch.arctan(By * x)))) + Sv

    def forward(self, X, U):

        vyc, psidt = torch.split(X, split_size_or_sections=1, dim=1)
        vxc, d1 = torch.split(U, split_size_or_sections=1, dim=1)
        
        d2 = torch.zeros(d1.shape).to(X.device)

        m = self.get_parameter('m')
        g = self.get_parameter('g')
        lf = self.get_parameter('lf')
        lr = self.get_parameter('l') - lf

        a = self.get_parameter('a')
        b = self.get_parameter('b')
        L = self.get_parameter('L')

        iz = self.get_parameter('iz')
      
        self.m_inv = 1 / m
        self.iz_inv = 1 / iz

        mf = m * g * lf/self.get_parameter('l')
        mr = m * g * lr/self.get_parameter('l')

        vy = vyc + (lf + a) * psidt
        vx = vxc + (L/2 + b) * psidt

        # Project on tire frame
        vxp1 = vx * torch.cos(d1) + (vy + lf * psidt) * torch.sin(d1)
        vxp2 = vx * torch.cos(d2) # + (vy - lr * psidt) * torch.sin(d2)
       
        vyp1 = (vy + lf * psidt) * torch.cos(d1) - vx * torch.sin(d1) # + b
        vyp2 = (vy - lr * psidt) * torch.cos(d2) # - vx * torch.sin(d2) # + b

        # Compute lateral slip angles
        alpha1 = self.get_slipping_angle(vxp1, vyp1, d1)
        alpha2 = self.get_slipping_angle(vxp2, vyp2, d2)

        # Compute lateral tire forces
        fyp1 = mf * self.get_lateral_force(alpha1, 1)
        fyp2 = mr * self.get_lateral_force(alpha2, 2)

        # Project on carbody frame
        self.fy1 = fyp1 * torch.cos(d1) 
        self.fy2 = fyp2 # * torch.cos(d2)

        vydt = self.m_inv * (self.fy1 + self.fy2) # - vx * torch.sin(d1) * psidt
        psidt2 = self.iz_inv * (lf * self.fy1 - lr * self.fy2)
        
        if self.output_format == 'acceleration':
            out = torch.concat((vydt, psidt2), dim=1)
        else:
            psidt = psidt2 * self.dt
            vy = vydt * self.dt - (lf + a) * psidt
            out = torch.concat((vy, psidt), dim=1)
        return out
    
    def get_parameters(self, standardize = False):
        parameters = {}
        # value : mean, std, gradient, not random
        for key, value in self.parameter['global'].items():
            if value[2]:
                parameters[key] = self.get_parameter(key, standardize).item()
        return parameters