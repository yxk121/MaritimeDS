from .base_model import BaseModel
from . import networks
from torchvision import transforms
import numpy as np
from data.base_dataset import get_transform
import torch
import torch.nn as nn
from PIL import Image

class TestModel(BaseModel):
    """ This TesteModel can be used to generate CycleGAN results for only one direction.
    This model will automatically set '--dataset_mode single', which only loads the images from one collection.

    See the test instruction for more details.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The model can only be used during test time. It requires '--dataset_mode single'.
        You need to specify the network using the option '--model_suffix'.
        """
        assert not is_train, 'TestModel cannot be used during training time'
        parser.set_defaults(dataset_mode='single')
        parser.add_argument('--model_suffix', type=str, default='', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        self.visual_names = ['real', 'fake']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G' + opt.model_suffix, 'EMI' + opt.model_suffix, 'EMT' + opt.model_suffix]  # only generator is needed.
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # netEMT
        transit_nc = 128
        transit_embedding_dim = 8
        self.netEMT = nn.Sequential(
            nn.Linear(transit_nc, 64),
            nn.ReLU(),
            nn.Linear(64, transit_embedding_dim),
            nn.ReLU()
        )
        # netEMI
        transit_nc = 17
        transit_embedding_dim = 1
        self.netEMI = nn.Sequential(
            nn.Linear(transit_nc, 8),
            nn.ReLU(),
            nn.Linear(8, transit_embedding_dim),
            nn.ReLU()
        )
        self.netEMT = networks.init_net(self.netEMT, init_type='normal', init_gain=0.02, gpu_ids=opt.gpu_ids)
        self.netEMI = networks.init_net(self.netEMI, init_type='normal', init_gain=0.02, gpu_ids=opt.gpu_ids)
        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        setattr(self, 'netG' + opt.model_suffix, self.netG)  # store netG in self.
        setattr(self, 'netEMI' + opt.model_suffix, self.netEMI)
        setattr(self, 'netEMT' + opt.model_suffix, self.netEMT)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """

        self.combine = input['A'].to(self.device)  # 1,139,256,256 tensor
        self.combine = self.combine.reshape([139, 256, 256])
        traj, transist = self.combine.split([11, 128], dim=0)  # [11, 256, 256];[128, 256, 256]
        pl, sd = traj.split([2, 9], dim=0)  # point&line;speed&direction tensor [2, 256, 256];[9, 256, 256]
        # pl = pl.reshape([-1, 256, 256, 2]).permute(0, 3, 1, 2).to(self.device)
        pl = pl.reshape([-1, 2, 256, 256]).to(self.device)  # 2,256,256->1,2,256,256

        transist = transist.permute(1, 2, 0).to(self.device)  # 256,256,128
        embedded_trans = self.netEMT(transist.reshape([-1, 256, 256, 128])).to(self.device)  # 1,256,256,8
        # sd = sd.reshape([-1, 256, 256, 9]).to(self.device)
        sd = sd.reshape([-1, 9, 256, 256]).permute(0, 2, 3, 1).to(self.device)  # 9,256,256->1,256,256,9
        # (1, 256, 256, 17)
        features = torch.cat([sd, embedded_trans], 3).to(self.device)  # 1, 256, 256, 9+8
        embedded_features = self.netEMI(features).permute(0, 3, 1, 2).to(self.device)  # 1,1,256,256
        # traj = torch.from_numpy(traj).reshape([-1, 256, 256, 11])  # 256,256,11->1,256,256,11
        gtimg = torch.cat([pl, embedded_features], 1)  # 1,3,256,256

        #gtimg = gtimg.reshape([3, 256, 256]).permute(1, 2, 0).cpu()
        #transform1 = transforms.ToPILImage(mode="RGB")
        #A_img = transform1(np.array(gtimg.detach().numpy()))

        gtimg = gtimg.reshape([3, 256, 256]).permute(1, 2, 0).cpu()
        gtimg = gtimg.detach().numpy()
        gtimg = (gtimg - np.min(gtimg)) * 255 / np.max(gtimg)
        A_img = Image.fromarray(np.uint8(gtimg))

        transform_A = get_transform(self.opt, grayscale=False)
        self.real = transform_A(A_img).to(self.device)
        self.real = torch.unsqueeze(self.real, 0).to(self.device)
        #self.real = input['A'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass."""
        self.fake = self.netG(self.real)  # G(real)

    def optimize_parameters(self):
        """No optimization for test model."""
        pass
