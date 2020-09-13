import torch
from .cut_model import CUTModel


class SinCUTModel(CUTModel):
    """ This class implements the single image translation model (Fig 9) of
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = CUTModel.modify_commandline_options(parser, is_train)
        parser.add_argument('--lambda_R1', type=float, default=1.0,
                            help='weight for the R1 gradient penalty')
        parser.add_argument('--lambda_identity', type=float, default=1.0,
                            help='the "identity preservation loss"')

        parser.set_defaults(nce_includes_all_negatives_from_minibatch=True,
                            dataset_mode="singleimage",
                            netG="stylegan2",
                            stylegan2_G_num_downsampling=1,
                            netD="stylegan2",
                            gan_mode="nonsaturating",
                            num_patches=1,
                            nce_layers="0,2,4",
                            lambda_NCE=4.0,
                            ngf=10,
                            ndf=8,
                            lr=0.002,
                            beta1=0.0,
                            beta2=0.99,
                            load_size=1024,
                            crop_size=64,
                            preprocess="zoom_and_patch",
        )

        if is_train:
            parser.set_defaults(preprocess="zoom_and_patch",
                                batch_size=16,
                                save_epoch_freq=1,
                                save_latest_freq=20000,
                                n_epochs=8,
                                n_epochs_decay=8,

            )
        else:
            parser.set_defaults(preprocess="none",  # load the whole image as it is
                                batch_size=1,
                                num_test=1,
            )
            
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        if self.isTrain:
            if opt.lambda_R1 > 0.0:
                self.loss_names += ['D_R1']
            if opt.lambda_identity > 0.0:
                self.loss_names += ['idt']

    def compute_D_loss(self):
        self.real_B.requires_grad_()
        GAN_loss_D = super().compute_D_loss()
        self.loss_D_R1 = self.R1_loss(self.pred_real, self.real_B)
        self.loss_D = GAN_loss_D + self.loss_D_R1
        return self.loss_D

    def compute_G_loss(self):
        CUT_loss_G = super().compute_G_loss()
        self.loss_idt = torch.nn.functional.l1_loss(self.idt_B, self.real_B) * self.opt.lambda_identity
        return CUT_loss_G + self.loss_idt

    def R1_loss(self, real_pred, real_img):
        grad_real, = torch.autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True, retain_graph=True)
        grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
        return grad_penalty * (self.opt.lambda_R1 * 0.5)
