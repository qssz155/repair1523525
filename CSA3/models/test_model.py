from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks
import torch


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'


    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)

        self.mask_global = torch.ByteTensor(1, 1, opt.fineSize, opt.fineSize)


        self.mask_global.zero_()
        self.mask_global[:, :, int(self.opt.fineSize/4) + self.opt.overlap : int(self.opt.fineSize/2) + int(self.opt.fineSize/4) - self.opt.overlap,\
                                int(self.opt.fineSize/4) + self.opt.overlap: int(self.opt.fineSize/2) + int(self.opt.fineSize/4) - self.opt.overlap] = 1

        self.mask_type = opt.mask_type
        self.gMask_opts = {}

        if len(opt.gpu_ids) > 0:
            self.use_gpu = True
            self.mask_global = self.mask_global.cuda()

        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)

        #self.netG = networks.define_G(opt.input_nc, opt.output_nc,
                                      #opt.ngf, opt.which_model_netG, opt,self.mask_global,
                                      #opt.norm, opt.use_dropout,
                                      #opt.init_type,
                                      #self.gpu_ids)


        self.netG,self.Cosis_list,self.Cosis_list2, self.CSA_model= networks.define_G(opt.input_nc_g, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt, self.mask_global, opt.norm, opt.use_dropout, opt.init_type, self.gpu_ids, opt.init_gain)
        which_epoch = opt.which_epoch
        self.load_network(self.netG, 'G', opt.which_epoch)

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, input):
        # we need to use single_dataset mode
        input_A = input['A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.image_paths = input['A_paths']

    def test(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG(self.real_A)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def load(self, epoch):
        self.load_network(self.netG, 'G', epoch)
