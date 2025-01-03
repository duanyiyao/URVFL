import numpy as np
from torch.utils.data import Dataset

import sys,os
import random

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from baseline_module import *
from attack_module import *


from cifar_data_pre import *
from models import *
from grna_gia_img import *
from util_cifar import *


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup random seed
    seed = random.randint(10, 1000)
    print(f"Using seed: {seed}")
    set_seed(seed)
    
    # Setup paths
  
    dataset_path, save_model_path, results_path = setup_paths()
    
    # Load configuration
    config = get_config(dataset_path / 'config_cifar.json')
    
    clients_num = 2
    data_owners = ["client_"+ str(i+1) for i in range(clients_num)]

    train_batchsize = config['train_btz']
    small_batchsize = config['train_btz']
    test_batchsize = config['test_btz']

    trainloader, testloader, smallloader = data_pre(data_owners, device, train_batchsize, test_batchsize, small_batchsize, num_samples = config['aux_size'])
   
    
    encoder = ClientNet(34, 10)
    encoder_opt= torch.optim.Adam(encoder.parameters(), lr=1e-4)
    shadow_model = ClientNet(34, 10)
    shadow_opt= torch.optim.Adam(shadow_model.parameters(), lr=1e-4)
    server = ServerNet(34, 10)
    server_opt = torch.optim.Adam(server.parameters(), lr=1e-4, amsgrad=True)
    generator = HalfWidthGenerator()
    gene_opt = torch.optim.Adam(generator.parameters(), lr=1e-4)
    target_model = ClientNet(34, 10)
    target_opt = torch.optim.Adam(target_model.parameters(), lr=1e-4)
    disc1 = Discriminator1()
    disc1_opt = torch.optim.Adam(disc1.parameters(), lr=1e-3, amsgrad=True)

    disc2 = Discriminator2()
    disc2_opt = torch.optim.Adam(disc2.parameters(), lr=1e-3, amsgrad=True)

 
    save_seed = 319 ## your previous saved model's seed 
    if config['vfl_save']:
        pretrain_vfl(encoder, target_model, server, trainloader, testloader, encoder_opt, target_opt, server_opt, device, config, path, seed)
    else:
        encoder.load_state_dict(torch.load(save_model_path / f"{save_seed}vfl_enc.pt"))
        target_model.load_state_dict(torch.load(save_model_path / f"{save_seed}vfl_target.pt"))
        server.load_state_dict(torch.load(save_model_path / f"{save_seed}vfl_server.pt"))

    mse_re = grna(generator, gene_opt, encoder, target_model, server, smallloader, testloader, device, config)
        

    res = np.array([mse_re])
    np.save(results_path / f"grna_{seed}", res)
    print(f"save path : {seed}")
    
    
    
    
if __name__ == "__main__":
    main()

