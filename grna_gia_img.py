import os,sys
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm
from random import choice
from tqdm import tqdm
import random
import torch.nn.functional as F
import math
from attack_module import * 
from baseline_module import *
import gc
            
        
    
def pretrain_vfl(enc, target_model, server, train_dataloader, test_dataloader, enc_opt, target_opt, server_opt, device, config, path, seed):
    
    adv = config['adv']
    target = config['target']
    best_val_loss = 0.0
    bce =nn.BCELoss()
    ce = nn.CrossEntropyLoss()
    enc = enc.to(device)
    target_model = target_model.to(device)
    server = server.to(device)
    losses = []
    epochs = config['pretrain_epochs']
    for epoch in range (epochs):
        running_loss = 0.0
        enc.train()
        target_model.train()
        server.train()
        for data, labels in train_dataloader:
            data[adv] = data[adv].to(device)
            data[target] = data[target].to(device)
            labels =labels.to(device)
            if config['dataset'] == 'credit':
              
                labels = labels.to(torch.float32).reshape(labels.size(0), -1)
            

            enc_opt.zero_grad()
            target_opt.zero_grad()
            server_opt.zero_grad()
            enc_output = enc(data[adv])
            target_out = target_model(data[target])
  
        
            con_input = torch.cat([enc_output.requires_grad_(), target_out.requires_grad_()], -1)
          
            pred = server(con_input)
            if config['dataset'] in ('credit'):
                loss = bce(pred, labels)
            else:
                loss = ce(pred, labels)

            loss.backward()
            enc_opt.step()
            target_opt.step()
            server_opt.step()

            running_loss += loss.item()
  
        epoch_loss = running_loss / len(train_dataloader)
        losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')
        score = get_score(enc, target_model, server, test_dataloader, device, config)
        if score > best_val_loss:
            best_val_loss = score
          
            torch.save(enc.state_dict(), path / f"{seed}vfl_enc.pt")
            torch.save(target_model.state_dict(), path / f"{seed}vfl_target.pt")
            torch.save(server.state_dict(), path / f"{seed}vfl_server.pt")
    return losses


def test_grna(testloader, generator, adv, target, device):
    generator.eval()
    mse = nn.MSELoss()
    sum_psnr = 0.0
    sum_ssim = 0.0
    sum_mse = 0
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    for x, label in testloader:
        x[target], x[adv] = x[target].to(device), x[adv].to(device)
        noise = torch.randn_like(x[target])
        fake_input2netG = torch.cat((x[adv], noise), dim= -1)
        xhat = generator(fake_input2netG) 
        loss = mse(xhat,x[target]).cpu().item()
        sum_psnr += psnr(xhat,x[target]).cpu().item()
        sum_ssim += ssim(xhat,x[target]).cpu().item()
        sum_mse += loss
    print(f'Average MSE of generator: {sum_mse/ len(testloader)}')
    return (sum_mse/ len(testloader)), (sum_psnr/ len(testloader)), (sum_ssim/ len(testloader))
        
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
    
    # Enable mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    accumulation_steps = 4  # Adjust this value based on your memory constraints
    
    for epoch in range(config['grna_epochs']):
        gene_loss = 0
        generator.train()
        for i, (x, label) in enumerate(smallloader):
            x[adv], x[target] = x[adv].to(device), x[target].to(device)
            noise = torch.randn_like(x[target])
            fake_input2netG = torch.cat((x[adv], noise), dim=-1)
            
            with torch.cuda.amp.autocast():
                xhat = generator(fake_input2netG).requires_grad_()
                true_out = forward_VFL(x[adv], x[target], encoder, target_model, server).detach()
                fake_out = forward_VFL(x[adv], xhat, encoder, target_model, server)
                loss = criterion(fake_out, true_out)
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(gene_opt)
                scaler.update()
                gene_opt.zero_grad()
            
            gene_loss += loss.detach()
            del fake_input2netG, true_out, fake_out, xhat, x, label, noise
            gc.collect()
            torch.cuda.empty_cache()
        
        print(f'Epoch: {epoch} Loss: {gene_loss / len(smallloader)}')
    
    mse_re, psnr_re, ssim_re = test_grna(testloader, generator, adv, target, device)     
    return mse_re, psnr_re, ssim_re
        

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
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
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
    psnr_error = 0
    ssim_error = 0
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
            psnr_error += psnr(optimized_x_pas, x[target]).cpu().item()
            ssim_error += ssim(optimized_x_pas, x[target]).cpu().item()
        #print(f'MSE of the reconstruct data {recon_error / count}')
    else:
        print(f'Average recon error {recon_error/ len(testloader)}')
        decoder_mse = recon_error/len(testloader)
        mean_psnr = psnr_error/len(testloader)
        mean_ssim = ssim_error/len(testloader)
        enc_mse, enc_cos = test_dis(shadow_model, target_model, testloader, target, device)
        
        return decoder_mse,mean_psnr, mean_ssim, enc_mse, enc_cos
        
        
        