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
    
    G = generator.Generator_PNCCGAN(z_dim=args.z_dim, image_size=args.image_size, img_channels=args.img_chennels, num_classes=args.num_classes).to(device)
    D = discriminator.Discriminator_PNCCGAN(image_size=args.image_size, channels=args.img_chennels).to(device)
    C = classifier.CNN_Classifier(ing_channels=args.img_channels, num_classes=args.num_classes)
    
    g_loss_fn = torch.nn.BCELoss()
    d_loss_fn = torch.nn.BCELoss()
    c_loss_fn = torch.nn.CrossEntropyLoss()
    
    g_optimizer = torch.optim.Adam(G.parameters(), lr= args.G_learning_rate)
    d_optimizer = torch.optim.Adam(D.parameters(), lr= args.D_learning_rate)
    c_optimizer = torch.optim.Adam(C.parameters(), lr= args.C_learning_rage)
    
    g_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer= g_optimizer,lr_lambda= lambda epoch: 0.65 ** epoch )
    d_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer= d_optimizer,lr_lambda= lambda epoch: 0.65 ** epoch )
    c_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer= c_optimizer,lr_lambda= lambda epoch: 0.65 ** epoch )
    
    ckpt_dir = './output/%s/checkpoints.pt' % args.model_name
    model_dir = './output/%s/result.pt' % args.model_name
    classifier_model_dir = './output/%s/classifier.pt' %args.model_name

    if not os.path.exists('./output/%s' % args.model_name):
        os.mkdir('./output/%s' % args.model_name)
    try:
        ckpt = torch.load(ckpt_dir)
        start_epoch = ckpt['epoch']
        D.load_state_dict(ckpt['D'])
        G.load_state_dict(ckpt['G'])
        C.load_state_dict(ckpt['C'])
        d_optimizer.load_state_dict(ckpt['d_optimizer'])
        g_optimizer.load_state_dict(ckpt['g_optimizer'])
        c_optimizer.load_state_dict(ckpt['c_optimizer'])
    except:
        print(' [*] No checkpoint!')
        start_epoch = 0
        
    z_sample = torch.randn(args.c_dim * 10, args.z_dim).to(device)
    c_sample = torch.tensor(np.concatenate([np.eye(args.c_dim)] * 10), dtype=z_sample.dtype).to(device)

    if args.classifier_trained:
        C = torch.load(classifier_model_dir)

        for epoch in range(start_epoch, args.target_epoch):
            D.train()
            G.train()
            C.eval()
            
            previous_class = torch.FloatTensor(args.c_dim,1).fill(1.0)

            for i,(x, classes) in enumerate(train_loader):
                
                valid = torch.autograd.Variable(torch.FloatTensor(x.size(0), 1).fill_(1.0), requires_grad=False)
                fake = torch.autograd.Variable(torch.FloatTensor(x.size(0), 1).fill_(0.0), requires_grad=False)
                class_ground = torch.FloatTensor(classes.size(0), 1).fill_(1.0)
                
                step = epoch * len(train_loader) + i + 1
                
                d_optimizer.zero_grad()
                g_optimizer.zero_grad()                

                x = x.to(device)
                z = torch.randn(args.batch_size, args.z_dim).to(device)
                c_x = torch.tensor(np.eye(args.c_dim)[classes.cpu().numpy()], dtype=z.dtype).to(device)
                classes_distribution = torch.FloatTensor(classes.size(0),1).fill_(1 / args.num_classes)
                c = previous_class
                
                gen_x = G(z, c)
                c_out = C(gen_x)
                previous_class = class_ground - torch.nn.Sigmoid(c_out)

                
                g_loss = g_loss_fn(D(gen_x), x) + c_loss_fn(C(gen_x), classes_distribution)
                
                g_loss.backward()
                g_optimizer.step()

                r_loss = d_loss_fn(D(x),valid)
                f_loss = d_loss_fn(D(gen_x), fake)
                d_loss = (r_loss + f_loss) / 2
                
                d_loss.backward()
                d_optimizer.step()

                
                writer.add_scalar('D/d_loss', d_loss.data.cpu().numpy(), global_step=step)
                writer.add_scalar('G/g_loss', g_loss.data.cpu().numpy(), global_step=step)
                
                print("Epoch: (%5d) step: (%5d/%5d)" %(epoch, i+1, len(train_loader)))

            g_scheduler.step()
            d_scheduler.step()

            torch.save({
                'epoch': epoch + 1,
                'D': D.state_dict(),
                'G': G.state_dict(),
                'D_opt': d_optimizer.state_dict(),
                'G_opt': g_optimizer.state_dict(),
                }, ckpt_dir)
        torch.save({
            'D': D.state_dict(),
            'G': G.state_dict(),
            'D_opt': d_optimizer.state_dict(),
            'G_opt': g_optimizer.state_dict(),
            }, model_dir)
    else:

        for epoch in range(start_epoch, args.target_epoch):
            D.train()
            G.train()
            C.train()
            
            previous_class = torch.FloatTensor(args.c_dim,1).fill(1.0)

            for i,(x, classes) in enumerate(train_loader):
                
                valid = torch.autograd.Variable(torch.FloatTensor(x.size(0), 1).fill_(1.0), requires_grad=False)
                fake = torch.autograd.Variable(torch.FloatTensor(x.size(0), 1).fill_(0.0), requires_grad=False)
                step = epoch * len(train_loader) + i + 1
                
                d_optimizer.zero_grad()
                g_optimizer.zero_grad()
                c_optimizer.zero_grad()    
                

                x = x.to(device)
                z = torch.randn(args.batch_size, args.z_dim).to(device)
                c_x = torch.tensor(np.eye(args.c_dim)[classes.cpu().numpy()], dtype=z.dtype).to(device)
                c = previous_class
                
                gen_x = G(z, c)
                
                g_loss = g_loss_fn(D(gen_x), x)
                
                g_loss.backward()
                g_optimizer.step()

                r_loss = d_loss_fn(D(x),valid)
                f_loss = d_loss_fn(D(gen_x), fake)
                d_loss = (r_loss + f_loss) / 2
                
                d_loss.backward()
                d_optimizer.step()

                c_out = C(gen_x)

                previous_class = c_out

                c_loss= c_loss_fn(c_out, c_x)

                c_loss.backward()
                c_optimizer.step()
                
                writer.add_scalar('D/d_loss', d_loss.data.cpu().numpy(), global_step=step)
                writer.add_scalar('G/g_loss', g_loss.data.cpu().numpy(), global_step=step)
                
                print("Epoch: (%5d) step: (%5d/%5d)" %(epoch, i+1, len(train_loader)))

            g_scheduler.step()
            d_scheduler.step()
            c_scheduler.step()

            torch.save({
                'epoch': epoch + 1,
                'D': D.state_dict(),
                'G': G.state_dict(),
                'D_opt': d_optimizer.state_dict(),
                'G_opt': g_optimizer.state_dict(),
                }, ckpt_dir)
            
        torch.save({
            'D': D.state_dict(),
            'G': G.state_dict(),
            'C': C.state_dict(),
            'D_opt': d_optimizer.state_dict(),
            'G_opt': g_optimizer.state_dict(),
            'C_opt': c_optimizer.state_dict()
            }, model_dir)

def trainning_CNN_Classifer(args):
    logger = utils.get_logger("PCNN GAN Training")
    if args.use_tensorboard_logging:
        writer = SummaryWriter(f"runs/{args.model_name}_logs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = dataset_init.get_dataset_loader(args, args.dataset)

    C = classifier.CNN_Classifier(args.img_channels, args.num_classes)

    C_loss_fn = torch.nn.CrossEntropyLoss()

    C_optimizer = torch.optim.Adam(C.parameters(), lr= args.C_learning_rate)

    C_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer = C_optimizer, lr_lambda=lambda epoch: 0.65 ** epoch)

    ckpt_dir = './output/%s/checkpoints.pt' % args.model_name
    classifier_model_dir = './output/%s/classifier.pt' %args.model_name

    for epoch in range(0, args.C_train_epoch):
        C.train()
        training_total_loss = 0.0
        for i, (img, label) in enumerate(train_loader):
            step = epoch * len(train_loader) + i + 1

            C_optimizer.zero_grad()

            x = img.to(device)
            output = C(x)

            C_loss = C_loss_fn(output, label)

            C_loss.backward()
            C_optimizer.step()

            training_total_loss += C_loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {training_total_loss / 2000:.3f}')
                training_total_loss = 0.0
    torch.save(C, classifier_model_dir)