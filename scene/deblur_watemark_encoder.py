import torch
from torch import nn

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
        'include_input' : True,
        'input_dims' : 3,
        'max_freq_log2' : multires-1,
        'num_freqs' : multires,
        'log_sampling' : True,
        'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim



class MLP(nn.Module):
    def __init__(
        self, input_ch, output_ch, hidden_ch, n_hidden_layers,
    ) :
        super(MLP, self).__init__()
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.hidden_ch = hidden_ch
        self.n_hidden_layers = n_hidden_layers

        self.layers = nn.ModuleList(
            [nn.Linear(input_ch, hidden_ch)] + [
                nn.Linear(hidden_ch, hidden_ch) for _ in range(n_hidden_layers)
            ] + [nn.Linear(hidden_ch, output_ch)]
        )

    def forward(self, x):
        for layer in self.layers[:-1] :
            x = layer(x)
            x = torch.relu(x)
        x = self.layers[-1](x)
        x = torch.sigmoid(x) + 1
        return x
    
    
class DeblurWatermarkEncoder(torch.nn.Module) :
    def __init__(
        self,
        scale_dim,
        rot_dim,
        msg_dim,
        pos_multires,
        dir_multires,
        hidden_ch,
        n_hidden_layers,
    ) :
        """
        assume that Dim(pos) == Dim(dir) == Dim(msg) == 3

        activation of scale, rot : sigmoid(out) + 1

        """
        super(DeblurWatermarkEncoder, self).__init__()
        self.pos_embed_fn, pos_input_ch = get_embedder(pos_multires)
        self.ray_embed_fn, ray_input_ch = get_embedder(dir_multires)
        
        self.mlp = MLP(
            input_ch    = pos_input_ch + ray_input_ch + scale_dim + rot_dim + msg_dim,
            output_ch   = scale_dim + rot_dim,
            hidden_ch   = hidden_ch, 
            n_hidden_layers = n_hidden_layers
        ).to('cuda')

    def forward(
        self,
        msg,
        camera_info,
        pos,
        scale,
        rot
    ) :
        g_to_v_vec = camera_info.camera_center - pos
        g_to_v_vec = g_to_v_vec / g_to_v_vec.norm()
        pos_embed = self.pos_embed_fn(pos)
        ray_embed = self.ray_embed_fn(g_to_v_vec)
        '''
        print(pos_embed.shape, pos_embed.device)
        print(ray_embed.shape, ray_embed.device)
        print(scale.shape, scale.device)
        print(rot.shape, rot.device)
        print(msg.shape, msg.device)
        '''
        # print("pos_embed : ", pos_embed)
        # print("ray_embed : ", ray_embed)
        # print("scale : ", scale)
        # print("rot : ", rot)
        # print("msg : ", msg)
        x = torch.cat([pos_embed, ray_embed, scale, rot, msg], dim=-1)
        return self.mlp(x)

'''
def get_embedder(
        multires,
        i=0,
        include_input=True,
    ):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : include_input,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class MLP(torch.nn.Module) :
    """
    3 Layer MLP
    input : (
        x : (60) positional encoded xyz position
        v : (60) positional encoded view direction
        r : (4) quaternion.
        s : (3) scale. 
    )
    xavier initialization
    """
    def __init__(
        self,
        input_dim : int,
        output_dim : int,
        num_layers : int = 4,
        hidden_dims : int = 64,

    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers

        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(self.input_dim, self.hidden_dims))
        for _ in range(self.num_layers-2) :
            self.layers.append(torch.nn.Linear(self.hidden_dims, self.hidden_dims))
        self.layers.append(torch.nn.Linear(self.hidden_dims, self.output_dim))

    def forward(
        self,
        x : torch.Tensor
    ) -> torch.Tensor :
        for layer in self.layers :
            x = torch.nn.functional.relu(layer(x))
        return x

class DeblurCoefficientHandler(torch.nn.Module) :
    """
    Gaussian의 scale, rotation 파라미터에 곱해줄 수를 계산하는 MLP 를 정의함.
    view direction 을 구함.
    position, view direction 에 대해 positional encoding 수행
    MLP에 통과시켜 scale, rotation 파라미터에 곱할 coefficient를 계산함.
    """
    def __init__(
        self,
        multires : int = 10,
        include_input_in_positional_encoding : bool = True,
        output_dims : int = 7,
    ) :
        super().__init__()
        self.embedder, output_dims = get_embedder(
            multires,
            i=0,
            include_input=include_input_in_positional_encoding
        )
        mlp_input_dim = output_dims * 2 + 4 + 3
        self.mlp = MLP(
            input_dim=mlp_input_dim,
            output_dim=output_dims,
            num_layers=4,
            hidden_dims=64
        )

    def forward(
        self,
        camera_info : torch.Tensor,
        pos : torch.Tensor,
        rotation : torch.Tensor,
        scale : torch.Tensor,
    ) :
        g_to_v_vec = camera_info.camera_center - pos
        g_to_v_vec = g_to_v_vec / g_to_v_vec.norm()
        
        pos_emb = self.embedder(pos)
        view_emb = self.embedder(g_to_v_vec)

        x = torch.cat([pos_emb, view_emb, rotation, scale], dim=-1)
        return self.mlp(x)
'''