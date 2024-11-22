import numpy as np
from torch.utils.data import Dataset
import sys,os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from baseline_module import *
from attack_module import *
from util_cifar import *
import random
from cifar_data_pre import *
from models import *
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json


def main():
   # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup random seed
    seed = random.randint(10, 1000)
    print(f"Using seed: {seed}")
    set_seed(seed)
    
    # Setup paths
  
    dataset_path, save_model_path, results_path = setup_paths()
    print(save_model_path)
    # Load configuration
    config = get_config(dataset_path / 'config_cifar.json')
    
    clients_num = 2
    data_owners = ["client_"+ str(i+1) for i in range(clients_num)]
    train_batchsize = config['train_btz']
    small_batchsize = config['train_btz']
    test_batchsize = config['test_btz']
    trainloader, testloader, smallloader = data_pre(data_owners, device, train_batchsize, test_batchsize, small_batchsize, num_samples = config['aux_size'])
   
    encoder = ClientNet(34, 10)
    encoder_opt= torch.optim.Adam(encoder.parameters(), lr=config['lr'])
    shadow_model = ClientNet(34, 10)
    shadow_opt= torch.optim.Adam(shadow_model.parameters(), lr=config['lr'])

    decoder = fullDecoder()
    decoder_opt = torch.optim.Adam(decoder.parameters(), lr=config['lr'])
    target_model = ClientNet(34, 10)
    target_opt = torch.optim.Adam(target_model.parameters(), lr=config['target_lr'])
    disc = Discriminator1()
    disc_opt = torch.optim.Adam(disc.parameters(), lr=config['disc_lr'])

    
    dis_ls, cosine_ls, mse_ls,psnr_ls, ssim_ls, losses = fsha_attack(shadow_model, encoder, target_model, decoder, disc, trainloader, testloader, smallloader,
                shadow_opt, encoder_opt, target_opt,  decoder_opt, disc_opt, device, config, save_model_path, seed)

    
    res = np.array([ dis_ls, cosine_ls, mse_ls,psnr_ls, ssim_ls])
    np.save(results_path / f"fsha_{seed}", res)
    print(f"save path : {seed}")
    
    
if __name__ == "__main__":
    main()