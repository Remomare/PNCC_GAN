#https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py#L97

import os
import torch
import torchvision
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
    embedding = torch.nn.Embedding(args.num_classes, args.num_classes).to(device)
    
    G = generator.Generator_PNCCGAN(z_dim=args.z_dim, img_size=args.img_size,  img_channels=args.img_channels, num_classes=args.num_classes).to(device)
    D = discriminator.Discriminator_PNCCGAN(channels=args.img_channels, img_size=args.img_size).to(device)
    C = classifier.CNN_Classifier(img_channels=args.img_channels, num_classes=args.num_classes).to(device)
    
    g_loss_fn = torch.nn.BCELoss()
    d_loss_fn = torch.nn.BCELoss()
    c_loss_fn = torch.nn.CrossEntropyLoss()
    
    g_optimizer = torch.optim.Adam(G.parameters(), lr= args.G_learning_rate)
    d_optimizer = torch.optim.Adam(D.parameters(), lr= args.D_learning_rate)
    c_optimizer = torch.optim.Adam(C.parameters(), lr= args.C_learning_rate)
    
    g_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer= g_optimizer,lr_lambda= lambda epoch: 0.95 ** epoch )
    d_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer= d_optimizer,lr_lambda= lambda epoch: 0.95 ** epoch )
    c_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer= c_optimizer,lr_lambda= lambda epoch: 0.95 ** epoch )
    
    ckpt_dir = args.ckpt_dir + '/checkpoint.pt'
    classifier_model_dir =  args.classifier_model_dir +'/classifier.pt'
    model_dir = args.model_dir + 'result.pt'

    start_epoch = 0
    
    z_sample = torch.randn(args.num_classes * 10, args.z_dim).to(device)
    c_sample = torch.tensor(np.concatenate([np.eye(args.num_classes)] * 10), dtype=z_sample.dtype).to(device)

    previous_class = torch.FloatTensor(args.batch_size, args.num_classes).fill_(0.0).to(device)

    if args.classifier_trained:
        C = torch.load(classifier_model_dir)
        
        D.train()
        G.train()
        C.eval()
            
        for epoch in range(start_epoch, args.target_epoch):

            print(f" ephch {epoch} prev_class: {previous_class[0]}" )

            for i,(x, classes) in enumerate(train_loader):
                
                valid = torch.autograd.Variable(torch.FloatTensor(x.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
                fake = torch.autograd.Variable(torch.FloatTensor(x.shape[0], 1).fill_(0.0), requires_grad=False).to(device)
                class_ground = torch.FloatTensor(args.batch_size, args.num_classes).fill_(1.0).to(device)
                
                step = epoch * len(train_loader) + i + 1
                
                g_optimizer.zero_grad()                

                x = torch.autograd.Variable(x.type(torch.Tensor)).to(device)
                z = torch.autograd.Variable(torch.randn(args.batch_size, args.z_dim)).float().to(device)
                c_x = torch.autograd.Variable(classes.type(torch.LongTensor)).to(device)
                classes_distribution = torch.autograd.Variable(torch.FloatTensor(args.batch_size, args.num_classes).fill_(1 / args.num_classes), requires_grad=False).to(device)
                prev_class = previous_class.detach()
                
                gen_x = G(z, prev_class)
                c_out = C(gen_x)
                
                gen_class = torch.nn.functional.sigmoid(c_out)
                class_dist_prob = torch.nn.functional.sigmoid(classes_distribution)
                previous_class = gen_class/(step) + prev_class * (step - 1) / (step)
                
                g_loss = g_loss_fn(D(gen_x), valid) 
                c_loss = c_loss_fn(previous_class,  class_dist_prob)

                g_total_loss = g_loss + c_loss
                
                g_total_loss.backward()

                g_optimizer.step()

                d_optimizer.zero_grad()
                
                c_x_embed = embedding(c_x)
                r_loss = d_loss_fn(D(x), valid)
                f_loss = d_loss_fn(D(gen_x.detach()), fake)
                d_loss = (r_loss + f_loss) / 2
                
                d_loss.backward()
                d_optimizer.step()

                if args.use_tensorboard_logging:
                    writer.add_scalar('D/d_loss', d_loss.data.cpu().numpy(), global_step=step)
                    writer.add_scalar('G/g_loss', g_loss.data.cpu().numpy(), global_step=step)
                
                print("Epoch: (%5d) step: (%5d/%5d) g_loss: (%.5f) c_loss: (%.5f) d_loss: (%.5f) " %(epoch, i+1, len(train_loader), g_loss, c_loss, d_loss))

                
                if (epoch * len(train_loader) + i) % 1500 == 1499:
                    torchvision.utils.save_image(gen_x.data[:25], "images/24/%d.png" % (epoch * len(train_loader) + i + 1), nrow=5, normalize=True)                
                
            g_scheduler.step()
            d_scheduler.step()

            """torch.save({
                'epoch': epoch + 1,
                'D': D.state_dict(),
                'G': G.state_dict(),
                'D_opt': d_optimizer.state_dict(),
                'G_opt': g_optimizer.state_dict(),
                }, ckpt_dir)"""
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
            
            previous_class = torch.FloatTensor(args.num_classes).fill(0)

            for i,(x, classes) in enumerate(train_loader):
                
                valid = torch.autograd.Variable(torch.FloatTensor(x.size(0), 1).fill_(1.0), requires_grad=False)
                fake = torch.autograd.Variable(torch.FloatTensor(x.size(0), 1).fill_(0.0), requires_grad=False)
                step = epoch * len(train_loader) + i + 1
                
                d_optimizer.zero_grad()
                g_optimizer.zero_grad()
                c_optimizer.zero_grad()    
                

                x = x.to(device)
                z = torch.randn(args.batch_size, args.z_dim).to(device)
                c_x = torch.tensor(np.eye(args.num_classes)[classes.cpu().numpy()], dtype=z.dtype).to(device)
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
            'D': D.state_dict(),
            'G': G.state_dict(),
            'C': C.state_dict(),
            'D_opt': d_optimizer.state_dict(),
            'G_opt': g_optimizer.state_dict(),
            'C_opt': c_optimizer.state_dict()
            }, model_dir)


def training_PNCC_GAN_no_class(args):
    
    logger = utils.get_logger("PCNN GAN Training")
    if args.use_tensorboard_logging:
        writer = SummaryWriter(f"runs/{args.model_name}_logs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = dataset_init.get_dataset_loader(args, args.dataset)
    
    G = generator.Generator_PNCCGAN_no_classembed(z_dim=args.z_dim, img_size=args.img_size, img_channels=args.img_channels, num_classes=args.num_classes).to(device)
    D = discriminator.Discriminator_PNCCGAN(img_size=args.img_size, channels=args.img_channels).to(device)
    C = classifier.CNN_Classifier(img_channels=args.img_channels, num_classes=args.num_classes).to(device)
    
    g_loss_fn = torch.nn.BCELoss()
    d_loss_fn = torch.nn.BCELoss()
    c_loss_fn = torch.nn.CrossEntropyLoss()
    
    g_optimizer = torch.optim.Adam(G.parameters(), lr= args.G_learning_rate)
    d_optimizer = torch.optim.Adam(D.parameters(), lr= args.D_learning_rate)
    c_optimizer = torch.optim.Adam(C.parameters(), lr= args.C_learning_rate)
    
    g_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer= g_optimizer,lr_lambda= lambda epoch: 0.65 ** epoch )
    d_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer= d_optimizer,lr_lambda= lambda epoch: 0.65 ** epoch )
    c_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer= c_optimizer,lr_lambda= lambda epoch: 0.65 ** epoch )
    
    ckpt_dir = args.ckpt_dir + '/checkpoint.pt'
    classifier_model_dir =  args.classifier_model_dir +'/classifier.pt'
    model_dir = args.model_dir + 'result.pt'

    start_epoch = 0
    
    z_sample = torch.randn(args.num_classes * 10, args.z_dim).to(device)
    c_sample = torch.tensor(np.concatenate([np.eye(args.num_classes)] * 10), dtype=z_sample.dtype).to(device)

    if args.classifier_trained:
        C = torch.load(classifier_model_dir)

        for epoch in range(start_epoch, args.target_epoch):
            D.train()
            G.train()
            C.eval()
            
            previous_class = torch.FloatTensor(args.batch_size, args.num_classes).fill_(0.0).to(device)

            for i,(x, classes) in enumerate(train_loader):
                
                valid = torch.autograd.Variable(torch.FloatTensor(x.size(0), 1).fill_(1.0), requires_grad=False).to(device)
                fake = torch.autograd.Variable(torch.FloatTensor(x.size(0), 1).fill_(0.0), requires_grad=False).to(device)
                class_ground = torch.FloatTensor(args.batch_size, args.num_classes).fill_(1.0).to(device)
                
                step = epoch * len(train_loader) + i + 1
                
                g_optimizer.zero_grad()                

                x = torch.autograd.Variable(x.type(torch.Tensor)).to(device)
                z = torch.autograd.Variable(torch.randn(args.batch_size, args.z_dim)).float().to(device)
                c_x = torch.autograd.Variable(classes.type(torch.LongTensor))
                classes_distribution = torch.autograd.Variable(torch.FloatTensor(args.batch_size, args.num_classes).fill_(1 / args.num_classes)).to(device)
                c = previous_class
                
                gen_x = G(z, c)
                c_out = C(gen_x)
                
                gen_class = torch.nn.functional.sigmoid(c_out)
                original_class  = torch.nn.functional.sigmoid(C(x.detach()))
                previous_class = gen_class.detach()
                
                g_loss = g_loss_fn(D(gen_x), valid) 
                c_loss = c_loss_fn(gen_class,  classes_distribution) + c_loss_fn(original_class, c_x) 
                
                g_total_loss = g_loss + c_loss /2
                
                g_total_loss.backward()

                g_optimizer.step()


                d_optimizer.zero_grad()
                
                r_loss = d_loss_fn(D(x),valid)
                f_loss = d_loss_fn(D(gen_x.detach()), fake)
                d_loss = (r_loss + f_loss) / 2
                
                d_loss.backward()
                d_optimizer.step()

                if args.use_tensorboard_logging:
                    writer.add_scalar('D/d_loss', d_loss.data.cpu().numpy(), global_step=step)
                    writer.add_scalar('G/g_loss', g_loss.data.cpu().numpy(), global_step=step)
                
                if (epoch * len(train_loader) + i) % 10 == 9:
                    print("Epoch: (%5d) step: (%5d/%5d) g_loss: (%.5f) c_loss: (%.5f) d_loss: (%.5f)" %(epoch, i+1, len(train_loader), g_loss, c_loss, d_loss))

                if (epoch * len(train_loader) + i) % 1500 == 1499:
                    torchvision.utils.save_image(gen_x.data[:25], "images/%d.png" % (epoch * len(train_loader) + i + 1), nrow=5, normalize=True)
                
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
            
            previous_class = torch.FloatTensor(args.num_classes).fill(0)

            for i,(x, classes) in enumerate(train_loader):
                
                valid = torch.autograd.Variable(torch.FloatTensor(x.size(0), 1).fill_(1.0), requires_grad=False)
                fake = torch.autograd.Variable(torch.FloatTensor(x.size(0), 1).fill_(0.0), requires_grad=False)
                step = epoch * len(train_loader) + i + 1
                
                d_optimizer.zero_grad()
                g_optimizer.zero_grad()
                c_optimizer.zero_grad()    
                

                x = x.to(device)
                z = torch.randn(args.batch_size, args.z_dim).to(device)
                c_x = torch.tensor(np.eye(args.num_classes)[classes.cpu().numpy()], dtype=z.dtype).to(device)
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

    C = classifier.CNN_Classifier(args.img_channels, args.num_classes).to(device)

    C_loss_fn = torch.nn.CrossEntropyLoss()

    C_optimizer = torch.optim.Adam(C.parameters(), lr= args.C_learning_rate)

    C_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer = C_optimizer, lr_lambda=lambda epoch: 0.65 ** epoch)

    ckpt_dir = args.ckpt_dir + '/checkpoint.pt'
    classifier_model_dir =  args.classifier_model_dir +'/classifier.pt'

    for epoch in range(0, args.C_train_epoch):
        C.train()
        for i, (img, label) in enumerate(train_loader):
            step = epoch * len(train_loader) + i + 1

            C_optimizer.zero_grad()

            x = img.to(device)
            output = C(x)

            C_loss = C_loss_fn(output, label.to(device))

            C_loss.backward()
            C_optimizer.step()

            if i % 200 == 199:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {C_loss:.6f}')
                
        C_scheduler.step()
        
    torch.save(C, classifier_model_dir)
    
def trainning_vanilla_gan(args):
    logger = utils.get_logger("Vanilla GAN Training")
    if args.use_tensorboard_logging:
        writer = SummaryWriter(f"runs/{args.model_name}_logs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = dataset_init.get_dataset_loader(args, args.dataset)

    G = generator.Generator_GAN(args.z_dim, args.img_channels, args.img_size).to(device)
    D = discriminator.Discriminator_GAN(args.img_channels, args.img_size).to(device)

    adversarial_loss = torch.nn.BCELoss()

    optimizer_G = torch.optim.Adam(G.parameters(), lr= args.C_learning_rate)
    optimizer_D = torch.optim.Adam(D.parameters(), lr= args.D_learning_rate)

    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer = optimizer_G, lr_lambda=lambda epoch: 0.95 ** epoch)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer = optimizer_D, lr_lambda=lambda epoch: 0.95 ** epoch)

    ckpt_dir = args.ckpt_dir + '/checkpoint.pt'

    for epoch in range(0, args.target_epoch):
        G.train()
        D.train()
        
        for i, (img, label) in enumerate(train_loader):
            step = epoch * len(train_loader) + i + 1
            
            valid = torch.autograd.Variable(torch.FloatTensor(img.size(0), 1).fill_(1.0), requires_grad=False).to(device)
            fake = torch.autograd.Variable(torch.FloatTensor(img.size(0), 1).fill_(0.0), requires_grad=False).to(device)
            
            real_img = torch.autograd.Variable(img.type(torch.FloatTensor)).to(device)
            
            optimizer_G.zero_grad()

            z = torch.autograd.Variable(torch.FloatTensor(np.random.normal(0,1,(img.shape[0], args.z_dim)))).to(device)
            
            gen_img = G(z)
            
            g_loss = adversarial_loss(D(gen_img), valid)
            
            g_loss.backward()
            optimizer_G.step()
            
            optimizer_D.zero_grad()
            
            real_loss = adversarial_loss(D(real_img), valid)
            fake_loss = adversarial_loss(D(gen_img.detach()), fake)
            
            d_loss = (real_loss + fake_loss) / 2
            
            d_loss.backward()
            optimizer_D.step()

            if i % 200 == 199:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {step + 1:6d}] g_loss: {g_loss:.5f} d_loss: {d_loss:.5f}')
            
            if step % 1500 == 0:
                torchvision.utils.save_image(gen_img.data[:25], "images/vanillaGAN/%d.png" % (epoch * len(train_loader) + i + 1), nrow=5, normalize=True)
                
        scheduler_G.step()
        scheduler_D.step()


    