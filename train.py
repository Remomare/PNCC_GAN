import os
import torch
import argparse

from models import training

def main(args):
    
    if args.model_name == "PNCC_GAN":
        if args.classifier_training:
            training.trainning_CNN_Classifer(args)
        
        training.training_PNCC_GAN(args)
    
    if args.model_name == 'GAN':
        training.trainning_vanilla_gan(args)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name', default= 'PNCC_GAN',
                        type=str)
    parser.add_argument('--dataset', default= 'MNIST',
                        type=str)
    parser.add_argument('--num_workers', default= 4,
                        type= int)
    parser.add_argument('--target_epoch', default= 700,
                        type=int)
    parser.add_argument('--C_train_epoch', default= 15,
                        type=int)
    parser.add_argument('--batch_size', default= 64,
                        type=int)
    parser.add_argument('--G_learning_rate', default= 0.0001,
                        type=float)
    parser.add_argument('--D_learning_rate', default= 0.0001,
                        type=float)
    parser.add_argument('--C_learning_rate', default= 0.001,
                        type=float)
    parser.add_argument('--z_dim', default= 100, 
                        type= int)
    
    parser.add_argument('--classifier_training', default= False,
                        type= bool)
    parser.add_argument('--classifier_trained', default= True,
                        type= bool)
    
    parser.add_argument('--use_tensorboard_logging', default= False,
                        type= bool)
        
    
    args = parser.parse_args()
    
    if args.dataset == 'MNIST':
        parser.add_argument('--img_size', default= 28,
                            type= int)
        parser.add_argument('--img_channels', default= 1,
                            type= int)
        parser.add_argument('--num_classes', default=10,
                            type= int)
        
    if torch.cuda.is_available():
        parser.add_argument('--use_gpu', default= True,
                            type= bool)
    else:
        parser.add_argument('--use_gpu', default=False,
                            type= bool)
    
    parser.add_argument('--model_dir', default= './output/%s/model' % args.model_name,
                        type= str)
    parser.add_argument('--ckpt_dir', default='./output/%s/checkpoint' % args.model_name,
                        type= str)
    parser.add_argument('--classifier_model_dir', default='./output/%s/classifier' %args.model_name,
                        type= str)
    
    args = parser.parse_args()
    
    current_path = os.getcwd()
    
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.classifier_model_dir):
        os.makedirs(args.classifier_model_dir)
    
    main(args)