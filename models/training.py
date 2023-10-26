#https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py#L97

import os
import torch
import numpy as np

from torch.utils.tensorboard.writer import SummaryWriter

from models import generator, discriminator, classifier, model_utils
from datasets import dataset_init
from utils import utils


def training_PNCC_GAN(args):
    
    logger = utils.get_logger("PCNN GAN Training")
    if args.use_tensorboard_logging:
        writer = SummaryWriter(f"runs/{args.model_name}_logs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = dataset_init.get_dataset_loader(args, args.dataset)
    
    G = generator.Generator_PNCCGAN(z_dim=args.z_dim, image_size=args.image_size, img_channels=args.img_chennels, num_class=args.num_class).to(device)
    D = discriminator.Discriminator_PNCCGAN(image_size=args.image_size, channels=args.img_chennels).to(device)
    
    d_loss_fn, g_loss_fn = model_utils.get_losses_fn(args.loss_mode)
    
    g_optimizer = torch.optim.Adam(G.parameters(), lr= args.g_learning_rate)
    d_optimizer = torch.optim.Adam(D.parameters(), lr= args.d_learning_rate)
    
    ckpt_dir = './output/%s/checkpoints' % args.model_name
    os.mkdir(args.ckpt_dir)
    try:
        ckpt = torch.load(ckpt_dir)
        start_epoch = ckpt['epoch']
        D.load_state_dict(ckpt['D'])
        G.load_state_dict(ckpt['G'])
        d_optimizer.load_state_dict(ckpt['d_optimizer'])
        g_optimizer.load_state_dict(ckpt['g_optimizer'])
    except:
        print(' [*] No checkpoint!')
        start_epoch = 0
        
    z_sample = torch.randn(args.c_dim * 10, args.z_dim).to(device)
    c_sample = torch.tensor(np.concatenate([np.eye(args.c_dim)] * 10), dtype=z_sample.dtype).to(device)
    
    for epoch in range(start_epoch, args.target_epoch):
        for i,(x, c_dense) in enumerate(train_loader):
            step = epoch * len(train_loader) + i + 1
            D.train()
            G.train()
    
            x = x.to(device)
            z = torch.randn(args.batch_size, args.z_dim).to(device)
            
            
    
    torch.save({
        'epoch': epoch,
        'D': D.state_dict(),
        'G': G.state_dict(),
        'D_opt': d_optimizer.state_dict(),
        'G_opt': g_optimizer.state_dict(),
                
                
        })