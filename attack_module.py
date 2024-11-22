import torch.nn as nn
from distutils.log import error
import numpy as np
import torch
import os,sys
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from itertools import cycle
from tqdm import tqdm
import torch.nn.functional as F
import torch
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure



def evaluate_table(enc, dec, target_model, shadow_model, test_dataloader, device, config):
    enc.eval()
    dec.eval()
    target_model.eval()
    shadow_model.eval()
    adv= config['adv']
    target = config['target']
    shadow_mse = 0.0
    shadow_train = 0.0
    target_mse = 0.0
    target_psnr = 0.0
    target_ssim = 0.0
  
    mse = nn.MSELoss()
    with torch.no_grad():
        for data, label in test_dataloader:
            data[adv] = data[adv].to(device)
            data[target] = data[target].to(device)
            label = label.to(device)
            if config['dataset'] in ('credit', 'iot'):
                target_img = torch.cat((data[adv], data[target]), -1)
                mask = torch.zeros_like(data[target]).to(device)
                mask_input = torch.cat([data[adv], mask], -1)
            else:
                mask_input = data[adv]

            enc_output = enc(mask_input)
            shadow_output = shadow_model(data[target])
            target_output = target_model(data[target])

            shadow_input = torch.cat((enc_output, shadow_output), -1)
            target_input = torch.cat((enc_output, target_output), -1)
            
           
            local_recon = dec(shadow_input)
            target_recon = dec(target_input)
                
            if config['dec_out_full']:
                if config['dataset'] in ('mnist', 'cifar10'):
                    loss_local = mse(local_recon[:, :, :, config['dim']:], data[target]).item()
                    loss_target = mse(target_recon[:,:, :,  config['dim']:], data[target]).item()
                  
                else:
                    loss_local = mse(local_recon[:, config['dim']:], data[target]).item()
                    loss_target = mse(target_recon[:, config['dim']:], data[target]).item()
            
            else:
                loss_local = mse(local_recon, data[target]).item()
                loss_target = mse(target_recon, data[target]).item()
       
            
            target_mse += loss_target
            shadow_mse += loss_local
    
            del mask_input, shadow_output, target_output

            
    mean_mse_target = target_mse / len(test_dataloader)
    mean_shadow = shadow_mse/ len(test_dataloader)

    print(f'Test Mean: {mean_mse_target}, shadow model mse: {mean_shadow} ')
    return mean_mse_target, mean_shadow


def evaluate_img(enc, dec, target_model, shadow_model, test_dataloader, device, config):
    enc.eval()
    dec.eval()
    target_model.eval()
    shadow_model.eval()
    adv= config['adv']
    target = config['target']
    shadow_mse = 0.0
    shadow_train = 0.0
    target_mse = 0.0
    target_psnr = 0.0
    target_ssim = 0.0
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    mse = nn.MSELoss()
    with torch.no_grad():
        for data, label in test_dataloader:
            data[adv] = data[adv].to(device)
            data[target] = data[target].to(device)
            label = label.to(device)
            if config['dataset'] in ('credit', 'iot'):
                target_img = torch.cat((data[adv], data[target]), -1)
                mask = torch.zeros_like(data[target]).to(device)
                mask_input = torch.cat([data[adv], mask], -1)
            else:
                mask_input = data[adv]

            enc_output = enc(mask_input)
            shadow_output = shadow_model(data[target])
            target_output = target_model(data[target])

            shadow_input = torch.cat((enc_output, shadow_output), -1)
            target_input = torch.cat((enc_output, target_output), -1)
            
           
            local_recon = dec(shadow_input)
            target_recon = dec(target_input)
                
            if config['dec_out_full']:
                if config['dataset'] in ('mnist', 'cifar10', "tiny"):
                    loss_local = mse(local_recon[:, :, :, config['dim']:], data[target]).item()
                    loss_target = mse(target_recon[:,:, :,  config['dim']:], data[target]).item()
                    psnr_target =  psnr(target_recon[:, :, :, config['dim']:], data[target]).item()
                    ssim_target = ssim(target_recon[:, :, :, config['dim']:], data[target]).item()
                else:
                    loss_local = mse(local_recon[:, config['dim']:], data[target]).item()
                    loss_target = mse(target_recon[:, config['dim']:], data[target]).item()
                    psnr_target = psnr(target_recon[:, config['dim']:], data[target]).item()
                    ssim_target = ssim(target_recon[:, config['dim']:], data[target]).item()
            else:
                loss_local = mse(local_recon, data[target]).item()
                loss_target = mse(target_recon, data[target]).item()
                psnr_target = psnr(target_recon, data[target]).item()
                ssim_target = ssim(target_recon, data[target]).item()
            
            target_mse += loss_target
            shadow_mse += loss_local
            target_psnr += psnr_target
            target_ssim += ssim_target
            del mask_input, shadow_output, target_output

            
    mean_mse_target = target_mse / len(test_dataloader)
    mean_shadow = shadow_mse/ len(test_dataloader)
    mean_psnr = target_psnr/ len(test_dataloader)
    mean_ssim = target_ssim/ len(test_dataloader)
    print(f'Test Mean: {mean_mse_target}, Test PSNR: {mean_psnr},\n Test SSIM: {mean_ssim}, shadow model mse: {mean_shadow} ')
    return mean_mse_target, mean_psnr, mean_ssim, mean_shadow



def test_dis(shadow, targetmodel, testloader, target, device):

    with torch.no_grad():
        shadow.eval()
        targetmodel.eval()
        norm_dis = 0
        mse = torch.nn.MSELoss()
        cosine_distance = 0
        for data_ptr, labels in testloader:
         
            data_ptr[target] = data_ptr[target].to(device)
            #data_ptr[target], data_ptr[adv] = data_ptr[target].to(device), data_ptr[adv].to(device)
            embedding_1 = shadow(data_ptr[target])
            embedding_2 = targetmodel(data_ptr[target])
            norm_dis += mse(embedding_1, embedding_2).item()
            embedding_1 = F.normalize(embedding_1.view(embedding_1.size(0),-1),dim=1)
            
            embedding_2 = F.normalize(embedding_2.view(embedding_2.size(0),-1),dim=1)
            
            cosine_similarity = F.cosine_similarity(embedding_1, embedding_2)
            cosine_distance += (1 - cosine_similarity).mean().item()
            
            del embedding_1, embedding_2
        print(f'Average mse distance of embedding : {norm_dis/len(testloader)}, Average cosine distance : {cosine_distance/len(testloader)}')
  
        return norm_dis/len(testloader), cosine_distance/len(testloader)



def pretrain_model(enc, dec, shadow_model, train_dataloader, test_dataloader, enc_opt, dec_opt, shadow_opt, device, config, path, seed):
    mse = nn.MSELoss()
    adv = config['adv']
    target = config['target']
    best_val_loss = 100.0

    enc = enc.to(device)
    dec = dec.to(device)
    shadow_model = shadow_model.to(device)
    losses = []
    epochs = config['pretrain_epochs']
    for epoch in range (epochs):
        running_loss = 0.0
        enc.train()
        dec.train()
        shadow_model.train()
        for data, label in train_dataloader:
            data[adv] = data[adv].to(device)
            data[target] = data[target].to(device)
            label = label.to(device)
            target_img = torch.cat((data[adv], data[target]), -1)
            
            if config['dataset'] in ('credit', 'iot'):
                mask = torch.zeros_like(data[target]).to(device)
                mask_input = torch.cat([data[adv], mask], -1)
            else:
                mask_input = data[adv]

            enc_opt.zero_grad()
            dec_opt.zero_grad()
            shadow_opt.zero_grad()
            enc_output = enc(mask_input)
            shadow_out = shadow_model(data[target])
  
        
            con_input = torch.cat([enc_output.requires_grad_(),  shadow_out.requires_grad_()], -1)
            
            recon = dec(con_input)
            if config['dec_out_full']:
                loss = mse(recon, target_img)
            else:
                loss = mse(recon, data[target])

            loss.backward()
            enc_opt.step()
            dec_opt.step()
            shadow_opt.step()

            running_loss += loss.item()
        target_model = shadow_model    
        epoch_loss = running_loss / len(train_dataloader)
        losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')
        if config['dataset'] in ('credit', 'iot'):
            recon, _ = evaluate_table(enc, dec, target_model, shadow_model, test_dataloader, device, config)
        else:
            recon, _, _, _ = evaluate_img(enc, dec, target_model, shadow_model, test_dataloader, device, config)
        
        if recon < best_val_loss:
            best_val_loss = recon
            if config['save']:
                torch.save(enc.state_dict(), path / f"{seed}urvfl_enc.pt")
                torch.save(dec.state_dict(), path / f"{seed}urvfl_dec.pt")
                torch.save(shadow_model.state_dict(), path / f"{seed}urvfl_shadow.pt")
            else:
                pass
    return losses


def cls_pretrain(target_model, encoder, decoder, shadow_model, dis, trainloader, smallloader, testloader, 
         target_opt, disc_opt, device, config, path, seed):

    target = config['target']
    adv = config['adv']

    encoder = encoder.to(device)
    decoder= decoder.to(device)
    target_model = target_model.to(device)
    shadow_model = shadow_model.to(device)
    dis = dis.to(device)
    
    best_val_loss = 100
    smallloader_cycle = cycle(smallloader)

    mse = nn.MSELoss()
    bce = nn.BCELoss()
    ce = nn.CrossEntropyLoss()

    losses = []
    features_ls = []
    if config['dataset'] in ('credit', 'iot'):
        evaluate_table(encoder, decoder, target_model, shadow_model,testloader, device, config)
    else:
        evaluate_img(encoder, decoder, target_model, shadow_model,testloader, device, config)
    cosine_ls =[] #cosine distance
    dis_ls =[] #mse distance
    result_mse = [] # test reconstruction loss
    result_psnr = []
    result_ssim = []
    for epoch in range(config['epochs']):
        running_loss = 0
        disc_loss_sum = 0
        dis.train()
        target_model.train()
        shadow_model.eval()
        for data_ptr, labels in tqdm(trainloader):

            data_ptr[target], data_ptr[adv] = data_ptr[target].to(device), data_ptr[adv].to(device)
            labels = labels.to(device)
            
            target_opt.zero_grad()
            disc_opt.zero_grad()
            
            target_model.train()
            
        
       
            local_img, local_labels = next(smallloader_cycle) 
            local_img[target], local_img[adv] = local_img[target].to(device), local_img[adv].to(device)
            local_labels = local_labels.to(device)
        
            
            encoder_output = shadow_model(local_img[target])
                
            target_input = target_model(data_ptr[target])
 
            dis_target, cls_target = dis(target_input)



            if config['gantype'] == 'gan':
                target_label = torch.zeros_like(dis_target)
                loss_dis = bce(dis_target, target_label)
                loss_dis.backward(retain_graph = True)
        
                target_opt.step()
                target_opt.zero_grad()
        
                running_loss = running_loss + loss_dis.detach().cpu()
        
                true_dis, true_cls = dis(target_input.detach())
         
                fake_dis, fake_cls = dis(encoder_output.detach())
           
        #     disc_loss = bce(true_dis, true_label) + bce(fake_dis, fake_label) + ce(true_cls, labels*2) + ce(fake_cls, local_labels*2 + 1)
                disc_loss = bce(true_dis, torch.ones_like(true_dis)) + bce(fake_dis, torch.zeros_like(fake_dis)) 
                
                
                disc_loss.backward(retain_graph = True)
                disc_opt.step()
                disc_opt.zero_grad()
                disc_loss_sum += disc_loss.cpu().item()
    
            
            elif config['gantype'] == 'cls':

                loss_dis = ce(cls_target, labels*2 + 1)
                loss_dis.backward(retain_graph =True)
        
                target_opt.step()
                target_opt.zero_grad()
        
                running_loss = running_loss + loss_dis.detach().cpu()
        
                true_dis, true_cls = dis(target_input.detach())
                true_label = torch.ones_like(true_dis)
                fake_dis, fake_cls = dis(encoder_output.detach())
                fake_label = torch.zeros_like(fake_dis)
                disc_loss = ce(true_cls, labels*2) + ce(fake_cls, local_labels*2 + 1)
        
                
                disc_loss.backward()
                disc_opt.step()
                disc_opt.zero_grad()
                disc_loss_sum += disc_loss.cpu().item()
                

            else:
                raise ValueError("Invalid type")

        else:
            losses.append(running_loss)
            print(f'Epoch: {epoch} running loss: {running_loss / len(trainloader)} discriminator loos : {disc_loss_sum/ len(trainloader)}')
            if config['dataset'] in ('credit', 'iot'):
                re_mse, _ = evaluate_table(encoder, decoder, target_model, shadow_model, testloader, device, config)
            else:
                re_mse,re_psnr, re_ssim, _ = evaluate_img(encoder, decoder, target_model, shadow_model, testloader, device, config)
            distance, cos_dis = test_dis(shadow_model, target_model, testloader, target, device)
            
            if re_mse < best_val_loss:
                best_val_loss = re_mse
                if config['save']:
                    torch.save(target_model.state_dict(), path / f"{seed}urvfl_target.pt")
           
            dis_ls.append(distance)
            cosine_ls.append(cos_dis)
            result_mse.append(re_mse)
            if config['dataset'] in ('mnist', 'cifar10', 'tiny'):
                result_psnr.append(re_psnr)
                result_ssim.append(re_ssim)
    if config['dataset'] in ('credit', 'iot'):
        
        return dis_ls, cosine_ls, result_mse
    elif config['dataset'] in ('mnist', 'cifar10', 'tiny'):
        return dis_ls, cosine_ls, result_mse, result_psnr, result_ssim
    else:
        return -1




def cls_sync(target_model, encoder, decoder, shadow_model, dis, trainloader, smallloader, testloader, 
        enc_opt, dec_opt, target_opt, shadow_opt,disc_opt,  device, config, path, seed):
    smallloader_cycle = cycle(smallloader)
    dis_ls, cosine_ls, result_psnr, result_mse, result_ssim  = [], [], [], [], []
    best_val_loss = 100
    encoder = encoder.to(device)
    decoder= decoder.to(device)
    target_model = target_model.to(device)
    shadow_model = shadow_model.to(device)
    dis = dis.to(device)

    target = config['target']
    adv = config['adv']

    mse = nn.MSELoss()
    bce = nn.BCELoss()
    ce = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(config['epochs']):
        running_loss = 0
        decoder_loss = 0
    
        for data_ptr, labels in tqdm(trainloader):
            labels = labels.to(device)
            
            data_ptr[adv] = data_ptr[adv].to(device)
            data_ptr[target] = data_ptr[target].to(device)  
            encoder.train()
            decoder.train()
            target_model.train()
            shadow_model.train()
            dis.train()
       
            local_img, local_labels = next(smallloader_cycle) 
            local_labels = local_labels.to(device)
            local_img[target], local_img[adv] = local_img[target].to(device), local_img[adv].to(device)

            target_img = torch.cat((local_img[adv], local_img[target]), -1)
            if config['dataset'] in ('credit', 'iot'):
                mask = torch.zeros_like(local_img[target]).to(device)
                mask_input = torch.cat([local_img[adv], mask], -1)
            else:
                mask_input = local_img[adv]
            enc_opt.zero_grad()
            dec_opt.zero_grad()
            target_opt.zero_grad()
            shadow_opt.zero_grad()
            disc_opt.zero_grad()

            shadow_output = shadow_model(local_img[target]).requires_grad_()
            encoder_output = encoder(mask_input)
            
           
            recon = decoder(torch.cat([encoder_output, shadow_output],-1))
            
            if config['dec_out_full']:
                loss_auto = mse(recon, target_img)
            else:
                loss_auto = mse(recon, local_img[target])
        
            loss_auto.backward()
            
            enc_opt.step()
            dec_opt.step()     
            shadow_opt.step()
            
            enc_opt.zero_grad()
            dec_opt.zero_grad()
            shadow_opt.zero_grad()

            
            decoder_loss += loss_auto.detach().cpu()
            
            target_input = target_model(data_ptr[target])
            adv_dis, adv_cls = dis(target_input.requires_grad_())
            disc_adv_loss = ce(adv_cls, labels * 2 + 1)

            disc_adv_loss.backward(retain_graph = True)
            target_opt.step()
            target_opt.zero_grad()

            true_dis, true_cls = dis(target_input.detach())
  
            fake_dis, fake_cls = dis(shadow_output.detach())
     
            disc_loss = ce(true_cls, labels * 2) + ce(fake_cls, local_labels * 2 + 1)
            disc_loss.backward()
            disc_opt.step()
            disc_opt.zero_grad()
           
            running_loss = running_loss + disc_adv_loss.detach().cpu()         
           
        else:
            losses.append(running_loss)
            print(f'Epoch: {epoch} Decoder Loss: {decoder_loss / len(trainloader)}, model loss: {running_loss / len(trainloader)}')
            if config['dataset'] in ('credit', 'iot'):
                re_mse, _ = evaluate_table(encoder, decoder, target_model, shadow_model, testloader, device, config)
            else:
                re_mse,re_psnr, re_ssim, _ = evaluate_img(encoder, decoder, target_model, shadow_model, testloader, device, config)
            distance, cos_dis = test_dis(shadow_model, target_model, testloader, target, device)
            
            if re_mse < best_val_loss:
                best_val_loss = re_mse
                if config['save']:
                    torch.save(encoder.state_dict(), path / f"{seed}sync_enc.pt")
                    torch.save(decoder.state_dict(), path / f"{seed}sync_dec.pt")
                    torch.save(shadow_model.state_dict(), path / f"{seed}sync_shadow.pt")
                    torch.save(target_model.state_dict(), path / f"{seed}syn_target.pt")
                else:
                    pass
            
            dis_ls.append(distance)
            cosine_ls.append(cos_dis)
            result_mse.append(re_mse)
            if config['dataset'] in ('mnist', 'cifar10', 'tiny'):
                    result_psnr.append(re_psnr)
                    result_ssim.append(re_ssim)
    if config['dataset'] in ('credit', 'iot'):
        
        return dis_ls, cosine_ls, result_mse
    elif config['dataset'] in ('mnist', 'cifar10', 'tiny'):
        return dis_ls, cosine_ls, result_mse, result_psnr, result_ssim
    else:
        return -1


