import os,sys
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

import torch
import torch.nn as nn

from tqdm import tqdm
import random
import torch.nn.functional as F
import math
from attack_module import * 
from baseline_module import *
from grna_gia_img import pretrain_vfl
            

def test_grna(testloader, generator, adv, target, device):
    generator.eval()
    mse = nn.MSELoss()
    
    sum_mse = 0
    for x, label in testloader:
        x[target], x[adv] = x[target].to(device), x[adv].to(device)
        noise = torch.randn_like(x[target])
        fake_input2netG = torch.cat((x[adv], noise), dim= -1)
        xhat = generator(fake_input2netG) 
        loss = mse(xhat,x[target]).cpu().item()
        
        sum_mse += loss
    print(f'Average MSE of generator: {sum_mse/ len(testloader)}')
    return (sum_mse/ len(testloader))
        
def forward_VFL(x1, x2, encoder, target_model, server):

    view1 = encoder(x1).requires_grad_()
    view2 = target_model(x2).requires_grad_()
    server_input = torch.cat([view1, view2], -1)
    output = server(server_input)
    return output

def grna(generator, gene_opt, encoder, target_model, server, smallloader, testloader, device, config):
    server = server.to(device)
    server.eval()
    encoder = encoder.to(device)
    encoder.eval()
    target_model = target_model.to(device)
    target_model.eval()
    generator = generator.to(device)
    if config['dataset'] in ('credit'):
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    target = config['target']
    adv = config['adv']
    
    
    for epoch in range(config['grna_epochs']):
        gene_loss = 0
        generator.train()
        for x, label in smallloader:
            x[adv], x[target] = x[adv].to(device), x[target].to(device)
            gene_opt.zero_grad()
            noise = torch.randn_like(x[target])
       
            fake_input2netG = torch.cat((x[adv], noise), dim=-1)
  
            xhat = generator(fake_input2netG).requires_grad_()
            
            true_out = forward_VFL(x[adv], x[target], encoder, target_model, server).detach()
            fake_out = forward_VFL( x[adv], xhat, encoder, target_model, server)
            loss = criterion(fake_out, true_out) 
            loss.backward()
            gene_opt.step()
            gene_opt.zero_grad()
            gene_loss += loss.detach()
        else:
            print(f'Epoch: {epoch} Loss: {gene_loss / len(smallloader)}')
    else:
        mse_re = test_grna(testloader, generator, adv, target, device)     
        
        return mse_re
  
        

def gia_attack( server, encoder, target_model, shadow_model, shadow_opt, smallloader, testloader, device, 
               config):
    
    server = server.to(device)
    server.eval()
    target_model = target_model.to(device)
    target_model.eval()
    shadow_model = shadow_model.to(device)
    shadow_model.eval()
    
    encoder = encoder.to(device)
    target = config['target']
    adv = config['adv']
    
    if config['dataset'] in ('credit'):
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    for epoch in range(config['gia_epochs']):
        gene_loss = 0
        encoder.train()
        for x, label in smallloader:
            x[adv], x[target] = x[adv].to(device), x[target].to(device)
            shadow_opt.zero_grad()
            
            true_out = forward_VFL(x[adv], x[target], encoder, target_model, server)
            fake_out = forward_VFL(x[adv], x[target], encoder, shadow_model, server)
            
            
            loss = mse(fake_out, true_out) 
            loss.backward()
            shadow_opt.step()
            shadow_opt.zero_grad()
            gene_loss += loss.detach()
        else:
            print(f'Epoch: {epoch} Loss: {gene_loss / len(smallloader)}')
    
    recon_error = 0
    encoder.eval()
  
    for x, label in tqdm(testloader):
        x[adv], x[target] = x[adv].to(device), x[target].to(device)
      
        
        x_pas_hat = torch.zeros_like(x[target],requires_grad=True)
        optimizer = torch.optim.Adam([x_pas_hat], lr=1e-3, amsgrad=True)
        true_out = forward_VFL(x[adv], x[target], encoder, shadow_model, server)
        
        
        for t in range(config['gia_opt_round']):
            optimizer.zero_grad()  # Clear gradients for the next set of operations
            fake_out = forward_VFL(x[adv], x_pas_hat, encoder, shadow_model, server)
            loss = criterion(true_out, fake_out)  # Compute the loss
            loss.backward(retain_graph = True)  # Compute gradient
            optimizer.step()  # Update the estimated input x_pas_hat
            optimizer.zero_grad()
        optimized_x_pas = x_pas_hat.detach()  # Detach the tensor from the graph
        
        if math.isnan(mse(optimized_x_pas, x[target]).cpu()):
            pass
        else:
            recon_error += mse(optimized_x_pas, x[target]).cpu().item()
        #print(f'MSE of the reconstruct data {recon_error / count}')
    else:
        print(f'Average recon error {recon_error/ len(testloader)}')
        decoder_mse = recon_error/len(testloader)
        enc_mse, enc_cos = test_dis(shadow_model, target_model, testloader, target, device)
        
        return decoder_mse, enc_mse, enc_cos
        
        
        