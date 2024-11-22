from torch import nn, optim
import torch.nn.functional as F
import torch
import tqdm
import random
import numpy as np
from itertools import cycle
from baseline_module import *
from attack_module import *
from util import *
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure


       
def splitguard(server, encoder,target_model, shadow_model, decoder, discriminator, trainloader, testloader, small_loader, server_opt, encoder_opt, target_opt, shadow_opt, decoder_opt, discriminator_opt, device, config, type):

    mult = 5
    exp = 2

    # number of randomized labels in a fake batch
    b_fake = int(config['train_btz'] * config['sg_ratio'])

    # probability of sending a fake batch
    p_fake = 0.1

    # batch index at which splitguard starts running
    N = 20
    index = 0
    fakes, r_1, r_2, fake_indices, scores = [], [], [], [], []
    
    smallloader_cycle = cycle(small_loader)
    best_loss = 100.0
    adv = config['adv']
    target = config['target']
    
    server = server.to(device)
    encoder = encoder.to(device)
    shadow_model = shadow_model.to(device)
    target_model = target_model.to(device)
    decoder = decoder.to(device)
    discriminator = discriminator.to(device)
    attack = True
    
    for epoch in range(config['fsha_epochs']):
        D_loss = 0
        decoder_loss = 0
        model_loss = 0
        server.train()
        encoder.train()
        decoder.train()
        shadow_model.train()
        target_model.train()
        discriminator.train()

        for data_ptr, labels in tqdm(trainloader):
            data_ptr[adv] = data_ptr[adv].to(device)
            data_ptr[target] = data_ptr[target].to(device)
            labels = labels.to(device)
            local_img, local_labels = next(smallloader_cycle) 
            local_img[target], local_img[adv] = local_img[target].to(device), local_img[adv].to(device)
            local_labels = local_labels.to(device)
            index += 1
            send_fakes = index > N and random.random() <= p_fake
            if send_fakes:
                # randomize labels by adding random int module 10 (num. classes)
                rand_labels = (labels + 1 ) % 2
                labels = torch.cat((rand_labels[:b_fake], labels[b_fake:]))

            if type == 'fsha':
                client_grad, loss, loss_auto, loss_dis = fsha_module(shadow_model, encoder, target_model, decoder, discriminator, data_ptr, local_img,
                shadow_opt, encoder_opt, target_opt,  decoder_opt, discriminator_opt, attack, config)
                client_grad = client_grad.flatten()
                decoder_loss += loss_auto
                model_loss += loss
                D_loss += loss_dis
           
            elif type == 'urvfl':
                client_grad, loss, loss_dis = urvfl_module(shadow_model,target_model, discriminator, data_ptr,labels, local_img,local_labels,
                target_opt, discriminator_opt, attack, config)
                client_grad = client_grad.flatten()
             
                model_loss += loss
                D_loss += loss_dis
            elif type == 'sync':
                client_grad, loss, loss_auto, loss_dis = sync_module(shadow_model, encoder, target_model, decoder, discriminator, data_ptr,labels, local_img,local_labels,shadow_opt, encoder_opt, target_opt,  decoder_opt, discriminator_opt, attack, config)
                client_grad = client_grad.flatten()
                decoder_loss += loss_auto
                model_loss += loss
                D_loss += loss_dis
            elif type == 'agn':
                client_grad, loss, loss_dis = agn_module( encoder, target_model, decoder, discriminator, data_ptr, local_img,
            encoder_opt, target_opt,  decoder_opt, discriminator_opt, attack, config)
                client_grad = client_grad.flatten()
                model_loss += loss
                D_loss += loss_dis
          
            elif type == 'normal':
                client_grad, loss = split_module(server, encoder, target_model, server_opt, encoder_opt, target_opt, data_ptr, labels, config)
                client_grad = client_grad.flatten()
                model_loss += loss
            else:
                raise ValueError("Invalid type")
 
            if send_fakes:
                fakes.append(client_grad.cpu())
                fake_indices.append(index)
                if len(r_1) > 0 and len(r_2) > 0:
                    sg = sg_score(fakes, r_1, r_2, mult=mult, exp=exp, raw=False)
                    scores.append(sg)
        
                    
                # do not update client model
            else:
                scores.append(0)
                if index > N:
                    # randomly split between R1 and R2
                    if random.random() <= 0.5:
                        r_1.append(client_grad.cpu())
                    else:
                        r_2.append(client_grad.cpu())     
            if not attack:
                if type in ("agn", "agn_cls"):
                    if config['dataset'] in ('credit', 'iot'):
                        mean_mse_target = evaluate_AGN_table(encoder, target_model, decoder, testloader, adv, target, config, device)
                    else:
                        mean_mse_target, mean_psnr, mean_ssim = evaluate_AGN_img(encoder, target_model, decoder, testloader, adv, target, config, device)
                    
                    mse_sim, cosine_sim = 0, 0
                else:
                    mse_sim, cosine_sim = test_dis(shadow_model, target_model, testloader, target, device)
                    if config['dataset'] in ('credit', 'iot'):
                        mean_mse_target, _ = evaluate_table(encoder, decoder, target_model, shadow_model, testloader, device, config)
                    else:
                        mean_mse_target, mean_psnr, mean_ssim, mean_shadow = evaluate_img(encoder, decoder, target_model, shadow_model, testloader, device, config)                    
                    
                print(f'Epoch: {epoch} Decoder Loss: {decoder_loss / len(trainloader)}, D loss: {D_loss / len(trainloader)}, model loss: {model_loss / len(trainloader)}')
                if config['dataset'] in ('credit', 'iot'):
                    return sg_value, mse_sim, cosine_sim, mean_mse_target
                else: 
                    return sg_value, mse_sim, cosine_sim, mean_mse_target, mean_psnr, mean_ssim
            if config['sg_defense']:
                if len(scores) > 15:
                    sg_value = sum(scores[-15:])/15
                    if sg_value < 0.9:
                        attack = False
                        pass
            torch.cuda.empty_cache()
    return scores


    

def gradient_scru(server, encoder,target_model, shadow_model, decoder, discriminator, trainloader, testloader, small_loader, server_opt, 
                  encoder_opt, target_opt, shadow_opt, decoder_opt, discriminator_opt, device,  config, type):
    count = 0
    dif_category_mean = []
    dif_variance = []
    same_category_mean = []
    same_variance = []
    
    smallloader_cycle = cycle(small_loader)
    scores = []
    adv = config['adv']
    target = config['target']
    k = 5
    length = 1000
    step = 50
    server = server.to(device)
    encoder = encoder.to(device)
    shadow_model = shadow_model.to(device)
    target_model = target_model.to(device)
    decoder = decoder.to(device)
    discriminator = discriminator.to(device)
    attack = True
    index = 0
    for epoch in range(config['gs_epochs']):
        D_loss = 0
        decoder_loss = 0
        model_loss = 0

        encoder.train()
        decoder.train()
        shadow_model.train()
        target_model.train()
        discriminator.train()

        for data_ptr, labels in tqdm(trainloader):
            count += 1
            if count > 50 :
                index +=  1
            
            data_ptr[adv] = data_ptr[adv].to(device)
            data_ptr[target] = data_ptr[target].to(device)
            labels = labels.to(device)
            local_img, local_labels = next(smallloader_cycle) 
            local_img[target], local_img[adv] = local_img[target].to(device), local_img[adv].to(device)
            local_labels = local_labels.to(device)
            
            if type == 'fsha':
                client_grad, loss, loss_auto, loss_dis = fsha_module(shadow_model, encoder, target_model, decoder, discriminator, data_ptr, local_img,
                shadow_opt, encoder_opt, target_opt,  decoder_opt, discriminator_opt, attack, config)
                client_grad = client_grad
                decoder_loss += loss_auto
                model_loss += loss 
                D_loss += loss_dis
            elif type == 'urvfl':
                client_grad, loss, loss_dis = urvfl_module(shadow_model,target_model, discriminator, data_ptr,labels, local_img,local_labels,
                target_opt, discriminator_opt, attack, config)
                client_grad = client_grad
               
                model_loss += loss
                D_loss += loss_dis
            elif type == 'sync':
                client_grad, loss, loss_auto, loss_dis = sync_module(shadow_model, encoder, target_model, decoder, discriminator, data_ptr,labels, local_img,local_labels,shadow_opt, encoder_opt, target_opt,  decoder_opt, discriminator_opt, attack, config)
                client_grad = client_grad
                decoder_loss += loss_auto
                model_loss += loss
                D_loss += loss_dis
            elif type == 'agn':
                client_grad, loss, loss_dis = agn_module( encoder, target_model, decoder, discriminator, data_ptr, local_img,
            encoder_opt, target_opt,  decoder_opt, discriminator_opt, attack, config)
                client_grad = client_grad
                model_loss += loss
                D_loss += loss_dis
            elif type == 'normal':
                client_grad, loss = split_module(server, encoder, target_model, server_opt, encoder_opt, target_opt, data_ptr, labels, config)
                client_grad = client_grad
                model_loss += loss
                
            else:
                raise ValueError("Invalid type")
 
            get_grad_set(client_grad, labels, dif_category_mean, dif_variance, same_category_mean, same_variance)  
            
            if config['gs_defense']:
                if index == 50:
                    index = 0
                    similarity = np.array([dif_category_mean, same_category_mean, dif_variance, same_variance])
                    scores = detection_score_compute(similarity, step)
            
            if not attack:
                if type == 'agn':
                    if config['dataset'] in ('credit', 'iot'):
                        mean_mse_target = evaluate_AGN_table(encoder, target_model, decoder, testloader, adv, target, config, device)
                    else:
                        mean_mse_target, mean_psnr, mean_ssim = evaluate_AGN_img(encoder, target_model, decoder, testloader, adv, target, config, device)
                    
                    
                    mse_sim, cosine_sim = 0, 0
                else:
                    mse_sim, cosine_sim = test_dis(shadow_model, target_model, testloader, target, device)
                    
                    
                    
                    if config['dataset'] in ('credit', 'iot'):
                        mean_mse_target, _ = evaluate_table(encoder, decoder, target_model, shadow_model, testloader, device, config)
                    else:
                        mean_mse_target, mean_psnr, mean_ssim, mean_shadow = evaluate_img(encoder, decoder, target_model, shadow_model, testloader, device, config)                    
                    
                
                print(f'Epoch: {epoch} Decoder Loss: {decoder_loss / len(trainloader)}, D loss: {D_loss / len(trainloader)}, model loss: {model_loss / len(trainloader)}')
                res = np.array([dif_category_mean, same_category_mean, dif_variance, same_variance])
                if config['dataset'] in ('credit', 'iot'):
                    return scores, mse_sim, cosine_sim, mean_mse_target,res
                else: 
                    return scores, mse_sim, cosine_sim, mean_mse_target, mean_psnr, mean_ssim, res
                
            if len(scores) > k:
                sg_value = sum(scores[-k:])/k
                if sg_value < 0.7:
                   attack = False
                   pass
            torch.cuda.empty_cache()
    res = np.array([dif_category_mean, same_category_mean, dif_variance, same_variance])   
    return res


def fsha_module(shadow, encoder, target_model, decoder, disc, data_ptr, local_img,
                shadow_opt, encoder_opt, target_opt,  decoder_opt, disc_opt, attack, config):
    
   
    adv = config['adv']
    target = config['target']
    
    mse = nn.MSELoss()
    bce = nn.BCELoss()

    decoder_opt.zero_grad()
    encoder_opt.zero_grad()
    disc_opt.zero_grad()
    target_opt.zero_grad()
    shadow_opt.zero_grad()
            
    target_emb = target_model(data_ptr[target])
    target_img = torch.cat([local_img[adv], local_img[target]], -1)
    adv_output = encoder(local_img[adv]).requires_grad_()
    shadow_output = shadow(local_img[target]).requires_grad_()
    decoder_output = decoder(torch.cat([adv_output, shadow_output], -1))
            
    if config['dec_out_full']:
        loss_auto = mse(decoder_output, target_img)
    else:
        loss_auto = mse(decoder_output, local_img[target])

    
    loss_auto.backward(retain_graph=True)
    shadow_opt.step()
    encoder_opt.step()
    decoder_opt.step()

    decoder_opt.zero_grad()
    encoder_opt.zero_grad()
    shadow_opt.zero_grad()

    
            
            #shadow_output = shadow(local_img[target])
    pred = disc(target_emb)
    if config['wgan']:
        loss = torch.mean(pred)
    else:
        loss = bce(pred, torch.zeros_like(pred))
    target_emb.retain_grad()
    loss.backward()
    if attack:
        target_opt.step()
        target_opt.zero_grad()

    batch_grad = target_emb.grad.detach().clone()
            

    local_predict = disc(shadow_output.detach())
    true_predict = disc(target_emb.detach())

    if config['wgan']:
        loss_dis = torch.mean(local_predict) - torch.mean(true_predict)

    else:
        
        loss_dis = bce(local_predict,torch.zeros_like(local_predict))+ bce(true_predict,torch.ones_like(true_predict))

    loss_dis.backward()
    
    disc_opt.step()
    disc_opt.zero_grad()

    del adv_output,shadow_output, target_emb

    return batch_grad, loss.cpu().item(), loss_auto.cpu().item(), loss_dis.cpu().item()
            
            

def agn_module( encoder, target_model, decoder, disc, data_ptr, local_img,
                encoder_opt, target_opt,  decoder_opt, disc_opt, attack, config):
    adv = config['adv']
    target = config['target']
    
    mse = nn.MSELoss()
    bce = nn.BCELoss()
    
    
    decoder_opt.zero_grad()
    encoder_opt.zero_grad()
    target_opt.zero_grad()
    
    target_emb = target_model(data_ptr[target]).requires_grad_()
    adv_emb = encoder(data_ptr[adv]).requires_grad_()

    decoder_output = decoder(data_ptr[adv], adv_emb, target_emb)

    true_img = torch.cat((local_img[adv], local_img[target]), -1)
    

    local_predict = disc(decoder_output.requires_grad_())
    fake_labels = torch.ones_like(local_predict)
    loss_target = bce(local_predict,fake_labels)
    target_emb.retain_grad()
    loss_target.backward(retain_graph = True)
    if attack:
        target_opt.step()
   
    batch_grad = target_emb.grad.detach().clone()
    encoder_opt.step()
    decoder_opt.step()

    decoder_opt.zero_grad()
    encoder_opt.zero_grad()
    target_opt.zero_grad()


    true_d = disc(true_img.detach())
    true_d_labels = torch.ones_like(true_d)
    decoder_d = decoder_output.detach()
    fake_d = disc(decoder_d.detach())
    fake_d_labels = torch.zeros_like(fake_d)
    loss_d = bce(true_d, true_d_labels)+ bce(fake_d, fake_d_labels)

    loss_d.backward()
    
    disc_opt.step()
    disc_opt.zero_grad()
    return batch_grad, loss_target.cpu().item(), loss_d.cpu().item()

def sync_module(shadow_model, encoder, target_model, decoder, disc, data_ptr,labels, local_img,local_labels,
                shadow_opt, encoder_opt, target_opt,  decoder_opt, disc_opt, attack, config):
    adv = config['adv']
    target = config['target']
    decoder_opt.zero_grad()
    encoder_opt.zero_grad()
    disc_opt.zero_grad()
    target_opt.zero_grad()
    shadow_opt.zero_grad()
    mse = nn.MSELoss()
    bce = nn.BCELoss()
    ce = nn.CrossEntropyLoss()
    
    target_img = torch.cat((local_img[adv], local_img[target]), -1)
    if config['dataset'] in ('credit', 'iot'):
        mask = torch.zeros_like(local_img[target])
        mask_input = torch.cat([local_img[adv], mask], -1)
    else:
        mask_input = local_img[adv]
    encoder_opt.zero_grad()
    decoder_opt.zero_grad()
    target_opt.zero_grad()
    shadow_opt.zero_grad()
    disc_opt.zero_grad()

    shadow_output = shadow_model(local_img[target]).requires_grad_()
    encoder_output = encoder(mask_input)
            
        
    recon = decoder(torch.cat([encoder_output, config['target_weight'] * shadow_output],-1))
    
    if config['dec_out_full']:
        loss_auto = mse(recon, target_img)
    else:
        loss_auto = mse(recon, local_img[target])

    loss_auto.backward()
    
    encoder_opt.step()
    decoder_opt.step()     
    shadow_opt.step()
    
    encoder_opt.zero_grad()
    decoder_opt.zero_grad()
    shadow_opt.zero_grad()

    

    target_input = target_model(data_ptr[target])

    lamda = 0

    adv_dis, adv_cls = disc(target_input.requires_grad_())
    
    
    disc_adv_loss = ce(adv_cls, labels * 2 + 1)
    target_input.retain_grad()
    disc_adv_loss.backward(retain_graph = True)
    if attack:
        target_opt.step()
        target_opt.zero_grad()
    batch_grad = target_input.grad.detach().clone()

    true_dis, true_cls = disc(target_input.detach())

    fake_dis, fake_cls = disc(shadow_output.detach())

    disc_loss = ce(true_cls, labels * 2) + ce(fake_cls, local_labels * 2 + 1)
    disc_loss.backward()
    disc_opt.step()
    disc_opt.zero_grad()
    
       
    return batch_grad, disc_adv_loss.cpu().item(), loss_auto.cpu().item(), disc_loss.cpu().item()



def urvfl_module(shadow_model,target_model, disc, data_ptr,labels, local_img,local_labels,
                 target_opt, disc_opt, attack, config):
    adv = config['adv']
    target = config['target']
  
    disc_opt.zero_grad()
    target_opt.zero_grad()

    ce = nn.CrossEntropyLoss()
    
    encoder_output = shadow_model(local_img[target])
        
    target_input = target_model(data_ptr[target])

    dis_target, cls_target = disc(target_input.requires_grad_())
    loss_dis = ce(cls_target, labels*2 + 1)
    target_input.retain_grad()
    loss_dis.backward(retain_graph = True)
    if attack:
        target_opt.step()
        target_opt.zero_grad()
    batch_grad = target_input.grad.detach().clone()

   

    true_dis, true_cls = disc(target_input.detach())
    true_label = torch.ones_like(true_dis)
    fake_dis, fake_cls = disc(encoder_output.detach())
    fake_label = torch.zeros_like(fake_dis)
    disc_loss = ce(true_cls, labels*2) + ce(fake_cls, local_labels*2 + 1)

    
    disc_loss.backward()
    disc_opt.step()
    disc_opt.zero_grad()


    return batch_grad, loss_dis.cpu().item(), disc_loss.cpu().item()

def split_module(server, encoder, target_model, server_opt, enc_opt, target_opt, data_ptr, labels, config):
    if config['dataset'] == 'credit':
        criterion = nn.BCELoss()
        labels = labels.to(torch.float32).reshape(labels.size(0), -1)

    else:
        criterion = nn.CrossEntropyLoss()
    server_opt.zero_grad()
    enc_opt.zero_grad()
    target_opt.zero_grad()
    
    adv = config['adv']
    target = config['target']
    target_emb = target_model(data_ptr[target]).requires_grad_()
    server_input = torch.cat((encoder(data_ptr[adv]), target_emb ),-1)
 
    pred = server(server_input)

    target_emb.retain_grad()
    
    loss = criterion(pred, labels)
    loss.backward()
    server_opt.step()
    enc_opt.step()
    target_opt.step()
    batch_grad = target_emb.grad.detach().clone()
    return batch_grad, loss.cpu().item()




def agn_cls_module( encoder, target_model, decoder, disc, server, data_ptr, local_img, labels,
                encoder_opt, target_opt,  decoder_opt, disc_opt, server_opt, attack, config):
    adv = config['adv']
    target = config['target']
    
    if config['dataset'] == 'credit':
        criterion = nn.BCELoss()
        labels = labels.to(torch.float32).reshape(labels.size(0), -1)

    else:
        criterion = nn.CrossEntropyLoss()
   
    bce = nn.BCELoss()
    
    server_opt.zero_grad()
    decoder_opt.zero_grad()
    encoder_opt.zero_grad()
    target_opt.zero_grad()
    
    target_emb = target_model(data_ptr[target]).requires_grad_()
    adv_emb = encoder(data_ptr[adv]).requires_grad_()
    pred = server(torch.cat([adv_emb, target_emb], -1))
    decoder_output = decoder(data_ptr[adv], adv_emb, target_emb)
    true_img = torch.cat((local_img[adv], local_img[target]), -1)

    local_predict = disc(decoder_output.requires_grad_())

    loss_target = bce(local_predict,torch.ones_like(local_predict))

    loss_target = loss_target + 1e-3* criterion(pred, labels)

    target_emb.retain_grad()
    loss_target.backward(retain_graph = True)

    if attack:
        target_opt.step()
    
    batch_grad = target_emb.grad.detach().clone()
    encoder_opt.step()
    decoder_opt.step()
    server_opt.step()
    decoder_opt.zero_grad()
    encoder_opt.zero_grad()
    target_opt.zero_grad()


    true_d = disc(true_img.detach())
    true_d_labels = torch.ones_like(true_d)
    decoder_d = decoder_output.detach()
    fake_d = disc(decoder_d.detach())
    fake_d_labels = torch.zeros_like(fake_d)
    loss_d = bce(true_d, true_d_labels)+ bce(fake_d, fake_d_labels)

    loss_d.backward()
    
    disc_opt.step()
    disc_opt.zero_grad()
    del target_emb, adv_emb
    return batch_grad, loss_target.cpu().item(), loss_d.cpu().item()
