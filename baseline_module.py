from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch
from tqdm import tqdm

from attack_module import *
import numpy as np
from itertools import cycle
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure


def split_train(server, encoder, target_model, data_ptr, labels, config):
    if config['dataset'] == 'credit':
        criterion = nn.BCELoss()
        labels = labels.to(torch.float32).reshape(labels.size(0), -1)

    else:
        criterion = nn.CrossEntropyLoss()

    adv = config['adv']
    target = config['target']

    server_input = torch.cat((encoder(data_ptr[adv]), target_model(data_ptr[target]) ),-1)
 
    pred = server(server_input)
    
    loss = criterion(pred, labels)
    return loss

def get_score(encoder, target_model, server, testloader, device, config):
    with torch.no_grad():
        encoder.eval()
        target_model.eval()
        server.eval()
        correct = 0
        total = 0
        adv = config['adv']
        target = config['target']
        for data, labels in testloader:
            data[adv] = data[adv].to(device)
            data[target] = data[target].to(device)
            labels = labels.to(device)
            server_input = torch.cat((encoder(data[adv]), target_model(data[target]) ), -1)
            output = server(server_input)
            if config['dataset'] == 'credit':
                threshold = 0.5
                pred = (output >= threshold).long()
            else:
                pred = output.argmax(1)
            correct += pred.eq(labels.data.view_as(pred)).sum()
            total += len(labels)
        accuracy = (correct/total)
        print("Accuracy: {}".format(accuracy))
        return(accuracy * 100 )


#####FSHA#####



def fsha_attack(shadow, encoder, target_model, decoder, disc, trainloader, testloader, small_loader,
                shadow_opt, encoder_opt, target_opt, decoder_opt, disc_opt, device, config, model_path, seed):
    
    recon_record = []
    mse_record = []
    cosine_record = []
    losses = []
    psnr_record = []
    ssim_record = []
    smallloader_cycle = cycle(small_loader)
    best_loss = 100.0
    adv = config['adv']
    target = config['target']
    
    shadow = shadow.to(device)
    encoder = encoder.to(device)
    target_model = target_model.to(device)
    decoder = decoder.to(device)
    disc = disc.to(device)
    mse = nn.MSELoss()
    bce = nn.BCELoss()
    
    for epoch in range(config['fsha_epochs']):
        D_loss = 0
        decoder_loss = 0
        targetmodel_loss = 0
        
        encoder.train()
        decoder.train()
        shadow.train()
        target_model.train()
        disc.train()
        
        for data_ptr, labels in tqdm(trainloader):
            data_ptr[adv] = data_ptr[adv].to(device)
            data_ptr[target] = data_ptr[target].to(device)
            labels = labels.to(device)

            decoder_opt.zero_grad()
            encoder_opt.zero_grad()
            disc_opt.zero_grad()
            target_opt.zero_grad()
            shadow_opt.zero_grad()
            
            target_emb = target_model(data_ptr[target])

            local_img, local_labels = next(smallloader_cycle) 
            
            local_img[target], local_img[adv] = local_img[target].to(device), local_img[adv].to(device)
            local_labels = local_labels.to(device)
            
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
            decoder_loss += loss_auto.detach().cpu()
            
            pred = disc(target_emb.requires_grad_())
            if config['wgan']:
                loss = torch.mean(pred)
            else:
                target_label = torch.zeros_like(pred)
                loss = bce(pred, target_label)
            loss.backward()
            
            target_opt.step()
            target_opt.zero_grad()
            targetmodel_loss += loss.detach().cpu().item()
            
            local_predict = disc(shadow_output.detach())
            true_predict = disc(target_emb.detach())

            if config['wgan']:
                loss_dis = torch.mean(local_predict) - torch.mean(true_predict)
            else:
                loss_D = bce(local_predict, torch.zeros_like(local_predict))
                loss_D_real = bce(true_predict, torch.ones_like(true_predict))
                loss_dis = loss_D + loss_D_real

            loss_dis.backward()
            
            disc_opt.step()
            disc_opt.zero_grad()
            D_loss += loss_dis.detach().cpu()
            
            # Clear unused variables to free memory
            del target_img, adv_output, shadow_output, decoder_output, local_predict, true_predict
            
            # Empty cache to free memory
            torch.cuda.empty_cache()
        
        print(f'Epoch: {epoch} Decoder Loss: {decoder_loss / len(trainloader)}, model loss: {targetmodel_loss / len(trainloader)}')
        mse_sim, cosine_sim = test_dis(shadow, target_model, testloader, target, device)
        if config['dataset'] in ('credit', 'iot'):
            recon, _ = evaluate_table(encoder, decoder, target_model, shadow, testloader, device, config)
        else:
            recon, mean_psnr, mean_ssim, _ = evaluate_img(encoder, decoder, target_model, shadow, testloader, device, config)
        if config['fsha_save']:
            if recon < best_loss:
                best_loss = recon 
                torch.save(encoder.state_dict(), model_path / f"{seed}fsha_enc.pt")
                torch.save(decoder.state_dict(), model_path / f"{seed}fsha_dec.pt")
                torch.save(shadow.state_dict(), model_path / f"{seed}fsha_shadow.pt")
                torch.save(target_model.state_dict(), model_path / f"{seed}fsha_target.pt")
        
        recon_record.append(recon)
        mse_record.append(mse_sim)
        cosine_record.append(cosine_sim)   
       
        losses.append(targetmodel_loss / len(trainloader)) 
        if config['dataset'] in ('mnist', 'cifar10', 'tiny'):
            psnr_record.append(mean_psnr)
            ssim_record.append(mean_ssim)
    if config['dataset'] in ('credit', 'iot'):
        
        return mse_record, cosine_record, recon_record, losses
    elif config['dataset'] in ('mnist', 'cifar10', 'tiny'):
        return mse_record, cosine_record, recon_record, psnr_record, ssim_record, losses
    else:
        return -1

  
#####PCAT#######

def pcat_attack(shadow, encoder, target_model, decoder, server, trainloader, testloader, small_loader,
                shadow_opt, encoder_opt, target_opt,  decoder_opt, server_opt,  device, config, model_path, seed):

    mse_record = []
    cosine_record = []
    losses = []
    psnr_record = []
    ssim_record = []
    best_loss = 0.0
    adv = config['adv']
    target = config['target']  
    smallloader_cycle = cycle(small_loader)
    shadow = shadow.to(device)
    target_model = target_model.to(device)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    server = server.to(device)
    mse = nn.MSELoss()

   
    for epoch in range(config['pcat_epochs']):
        running_loss = 0
        shadow_loss = 0
        encoder.train()
        server.train()
        shadow.train()
        target_model.train()
        
        for data_ptr, labels in tqdm(trainloader):

            data_ptr[adv] = data_ptr[adv].to(device)
            data_ptr[target] = data_ptr[target].to(device)
            labels = labels.to(device)


            shadow_opt.zero_grad()
            encoder_opt.zero_grad()
            server_opt.zero_grad()
            target_opt.zero_grad()

            local_img, local_labels = next(smallloader_cycle) 
            local_img[target], local_img[adv] = local_img[target].to(device), local_img[adv].to(device)
            local_labels = local_labels.to(device)
            
            loss_local = split_train(server, encoder, shadow, local_img, local_labels, config)  
            loss_local.backward()
            shadow_opt.step()
            shadow_opt.zero_grad()
            shadow_loss += loss_local.detach().cpu()
        
            loss = split_train(server, encoder, target_model, data_ptr, labels, config)
            loss.backward()
            target_opt.step()
            encoder_opt.step()
            server_opt.step()
            encoder_opt.zero_grad()
            server_opt.zero_grad()
            target_opt.zero_grad()

            running_loss += loss.detach().cpu()
        
        else:
            
            print(f'Epoch: {epoch} Loss: {running_loss / len(trainloader)}, encoder loss: {shadow_loss / len(trainloader)}')
            mse_sim, cosine_sim = test_dis(shadow, target_model, testloader, target, device)
            score = get_score(encoder, target_model, server, testloader, device, config)
            
            if score > best_loss:
                best_loss = score 
                
                if config['pcat_save']:
                    torch.save(encoder.state_dict(), model_path / f"{seed}pcat_enc.pt")
                    torch.save(shadow.state_dict(), model_path / f"{seed}pcat_shadow.pt")
                    torch.save(target_model.state_dict(), model_path / f"{seed}pcat_target.pt")
                    torch.save(server.state_dict(), model_path / f"{seed}pcat_server.pt")
                else:
                    pass
            
            
            
            mse_record.append(mse_sim)
            cosine_record.append(cosine_sim)   
            
            losses.append(running_loss / len(trainloader)) 


    else:
        print("Start train decoder")
        shadow.load_state_dict(torch.load(model_path / f"{seed}pcat_shadow.pt"))
        encoder.load_state_dict(torch.load(model_path / f"{seed}pcat_enc.pt"))
        target_model.load_state_dict(torch.load(model_path / f"{seed}pcat_target.pt"))
        target_model.eval()
        shadow.eval()
        encoder.eval()
        
     
        best_recon = 100.0
        for epoch in range(config['pcat_dec_epochs']):
            decoder_loss = 0
            decoder.train()

            for batch_index, (local_img, local_labels) in enumerate(small_loader):


                decoder_opt.zero_grad()
                local_img[target], local_img[adv] = local_img[target].to(device), local_img[adv].to(device)
                local_labels = local_labels.to(device)
                target_img = torch.cat([local_img[adv], local_img[target]], -1)
                
                adv_output = encoder(local_img[adv])
                shadow_output = shadow(local_img[target])
                
                
                decoder_input = torch.cat([adv_output, shadow_output ], -1)
      
                decoder_output = decoder(decoder_input)
             
                if config['dec_out_full']:
                    loss_auto = mse(decoder_output, target_img)
                else:
                    loss_auto = mse(decoder_output, local_img[target])
                loss_auto.backward()
                decoder_opt.step()
                decoder_opt.zero_grad()
                decoder_loss += loss_auto
                del adv_output, shadow_output,decoder_input
            else:
                print(f'Epoch: {epoch} Loss: {decoder_loss / len(small_loader)}')
        else:
            if config['dataset'] in ('credit', 'iot'):
                recon, _ = evaluate_table(encoder, decoder, target_model, shadow, testloader, device, config)
            else:
                recon, mean_psnr, mean_ssim, _ = evaluate_img(encoder, decoder, target_model, shadow, testloader, device, config)
                
            if recon < best_recon:
                best_recon = recon 

                if config['pcat_save']:
                    torch.save(decoder.state_dict(), model_path / f"{seed}pcat_dec.pt")
                else:
                    pass
   
    if config['dataset'] in ('credit', 'iot'):
        return mse_record, cosine_record, recon, losses
    elif config['dataset'] in ('mnist', 'cifar10', 'tiny'):
        return mse_record, cosine_record, recon, mean_psnr, mean_ssim, losses
    else:
        return -1
                 
#####SDAR#####           


def sdar_attack(shadow, encoder, target_model, decoder, server, disc1, disc2, trainloader, testloader, small_loader,
                shadow_opt, encoder_opt, target_opt,  decoder_opt, server_opt, disc1_opt, disc2_opt, device, config, model_path, seed):
    recon_record = []
    mse_record = []
    cosine_record = []
    losses = []
    psnr_record = []
    ssim_record = []
    best_loss = 100.0
    smallloader_cycle = cycle(small_loader)
    shadow = shadow.to(device)
    target_model = target_model.to(device)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    server = server.to(device)
    disc1 = disc1.to(device)
    disc2 = disc2.to(device)
    
    adv = config['adv']
    target = config['target']  
    mse = nn.MSELoss()
    if config['dataset'] == "credit":
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()


    bce = nn.BCELoss()
    for epoch in range(config['sdar_epochs']):
        d1_loss = 0
        d2_loss = 0
        decoder_loss = 0
        shadow_loss = 0
        model_loss = 0

        shadow.train()
     
        disc1.train()
        disc2.train()
        for data_ptr, labels in tqdm(trainloader):
            data_ptr[adv] = data_ptr[adv].to(device)
            data_ptr[target] = data_ptr[target].to(device)
            labels = labels.to(device)

            target_model.train()
            encoder.train()
            server.train()

            decoder_opt.zero_grad()
            encoder_opt.zero_grad()
            disc1_opt.zero_grad()
            disc1_opt.zero_grad()
            shadow_opt.zero_grad()
            server_opt.zero_grad()
            target_opt.zero_grad()

            loss = split_train(server, encoder, target_model, data_ptr, labels, config)
            loss.backward()
            
            server_opt.step()
            server_opt.step()
            target_opt.step()

            encoder_opt.zero_grad()
            server_opt.zero_grad()
            target_opt.zero_grad()
            model_loss += loss.detach().cpu()
            
            
            ## attack
            target_emb = target_model(data_ptr[target]).detach()
            local_img, local_labels = next(smallloader_cycle) 
            
            server.eval()
            encoder.eval()
            target_model.eval()
            local_img[target], local_img[adv] = local_img[target].to(device), local_img[adv].to(device)
            local_labels = local_labels.to(device)
            target_img = torch.cat([local_img[adv], local_img[target]], -1)
            
            
            
            shadow_output = shadow(local_img[target])
            encoder_out = encoder(local_img[adv]).detach()
            decoder_input = torch.cat([encoder_out, shadow_output], -1)
            
            
            loss_local = split_train(server, encoder, shadow, local_img, local_labels, config)
            
            local_predict = disc1(shadow_output)
            if config['wgan']:
                loss_fake_d = torch.mean(local_predict)
            else:
                loss_fake_d = bce(local_predict,torch.ones_like(local_predict))
            
            loss_1 = loss_local + config['sdar_lamda1'] * loss_fake_d

            
            decoder_output = decoder(decoder_input)
            
            if config['dec_out_full']:
                loss_auto = mse(decoder_output, target_img)
            else:
                loss_auto = mse(decoder_output, local_img[target])
            recon_fake_d = disc2(decoder_output)
            if config['wgan']:
                loss_recon_fake = torch.mean(recon_fake_d)
            else:
            
                loss_recon_fake = bce(recon_fake_d, torch.ones_like(recon_fake_d))
            
            loss_2 = loss_auto + config['sdar_lamda2']* loss_recon_fake
            
            
            loss_1.backward(retain_graph = True)
            loss_2.backward(retain_graph = True)

            shadow_opt.step()
            shadow_opt.zero_grad()
            
            decoder_opt.step()
            decoder_opt.zero_grad()


            true_d = disc1(target_emb.detach())
            fake_d = disc1(shadow_output.detach())
            if config['wgan']:
                loss_d1 = torch.mean(true_d) - torch.mean(fake_d)
            else:
       
                loss_d1 = bce(true_d,torch.ones_like(true_d))+ bce(fake_d, torch.zeros_like(fake_d))
            
            decoder.eval()
            if config['dec_out_full'] :
                true_d2 = disc2(target_img)
            else:
                true_d2 = disc2(local_img[target])
            fake_d2 = disc2(decoder_output.detach())    
            
            if config['wgan']:
                loss_d2 = torch.mean(true_d2) - torch.mean(fake_d2)
       
            loss_d2 = bce(true_d2, torch.ones_like(true_d2))+ bce(fake_d2, torch.zeros_like(fake_d2))

            loss_d1.backward(retain_graph = True)
            loss_d2.backward(retain_graph = True)
            
            disc1_opt.step()
            disc1_opt.zero_grad()
            
            disc2_opt.step()
            disc2_opt.zero_grad()
            
            d1_loss += loss_d1.detach().cpu()
            d2_loss += loss_d2.detach().cpu()
            decoder_loss += loss_2.detach().cpu()
            shadow_loss += loss_1.detach().cpu()
            
            del  shadow_output,decoder_input
            
        else:
            print(f'Epoch: {epoch} Decoder Loss: {decoder_loss / len(trainloader)}, Shdow model loss: {shadow_loss / len(trainloader)}, \
                  normal training loss: {model_loss / len(trainloader)}, D1 loss: {d1_loss / len(trainloader)}, D2 loss : {d2_loss / len(trainloader)}')

            mse_sim, cosine_sim = test_dis(shadow, target_model, testloader, target, device)
            if config['dataset'] in ('credit', 'iot'):
                recon, _ = evaluate_table(encoder, decoder, target_model, shadow, testloader, device, config)
            else:
                recon, mean_psnr, mean_ssim, _ = evaluate_img(encoder, decoder, target_model, shadow, testloader, device, config)
                
            if config['sdar_save']:
                if recon < best_loss:
                    best_loss = recon 
       
                    torch.save(encoder.state_dict(), model_path / f"{seed}sdar_enc.pt")
                    torch.save(decoder.state_dict(), model_path / f"{seed}sdar_dec.pt")
                    torch.save(shadow.state_dict(), model_path / f"{seed}sdar_shadow.pt")
                    torch.save(target_model.state_dict(), model_path / f"{seed}sdar_target.pt")
                else:
                    pass
            
            
            
            recon_record.append(recon)
            mse_record.append(mse_sim)
            cosine_record.append(cosine_sim) 
            
            losses.append(shadow_loss / len(trainloader)) 
            if config['dataset'] in ('mnist', 'cifar10', 'tiny'):
                psnr_record.append(mean_psnr)
                ssim_record.append(mean_ssim)
    
    else:
        if config['dataset'] in ('credit', 'iot'):
            return mse_record, cosine_record, recon_record, losses
        elif config['dataset'] in ('mnist', 'cifar10', 'tiny'):
            return mse_record, cosine_record, recon_record, psnr_record, ssim_record, losses
        else:
            return -1
      
      
      
### AGN

def agn_attack( encoder, target_model, decoder, disc, trainloader, testloader, small_loader,
     encoder_opt, target_opt,  decoder_opt, disc_opt, device, config, model_path, seed):
    recon_record, psnr_record, ssim_record = [], [], []
    smallloader_cycle = cycle(small_loader)
    losses = []
    best_loss = 100.0
    adv = config['adv']
    target = config['target']
    
  
    encoder = encoder.to(device)
    target_model = target_model.to(device)
    decoder = decoder.to(device)
    disc = disc.to(device)
    mse = nn.MSELoss()
    bce = nn.BCELoss()
    for epoch in range(config['agn_epochs']):
        D_loss = 0
        decoder_loss = 0

        
        encoder.train()
        decoder.train()
        target_model.train()
        disc.train()
        
        for data_ptr, labels in tqdm(trainloader):
            data_ptr[adv] = data_ptr[adv].to(device)
            data_ptr[target] = data_ptr[target].to(device)
            labels = labels.to(device)
            

            decoder_opt.zero_grad()
            encoder_opt.zero_grad()
 
            target_opt.zero_grad()
    
            
            target_emb = target_model(data_ptr[target]).requires_grad_()
            adv_emb = encoder(data_ptr[adv]).requires_grad_()

            decoder_output = decoder(data_ptr[adv], adv_emb, target_emb)
        
            local_img, local_labels = next(smallloader_cycle) 
            local_img[target], local_img[adv] = local_img[target].to(device), local_img[adv].to(device)
            true_img = torch.cat((local_img[adv], local_img[target]), -1)
            

            local_predict = disc(decoder_output.requires_grad_())
            fake_labels = torch.ones_like(local_predict)
            loss_target = bce(local_predict,fake_labels)
            loss_target.backward(retain_graph = True)
            target_opt.step()
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
            D_loss += loss_d.detach().cpu()
            decoder_loss += loss_target.detach().cpu()
            del target_emb, adv_emb, true_img
        else:
            print(f'Epoch: {epoch} Decoder Loss: {decoder_loss / len(trainloader)}, disc loss: {D_loss / len(trainloader)}')
            if config['dataset'] in ('credit', 'iot'):
                recon = evaluate_AGN_table(encoder, target_model, decoder, testloader, adv, target, config, device)
            else:
                recon, psnr_mean, ssim_mean = evaluate_AGN_img(encoder, target_model, decoder, testloader, adv, target, config, device)
            recon_record.append(recon)
            losses.append(decoder_loss/len(trainloader))
            if config['agn_save']:
                if recon < best_loss:
                    best_loss = recon 
                    torch.save(encoder.state_dict(), model_path / f"{seed}agn_enc.pt")
                    torch.save(decoder.state_dict(), model_path / f"{seed}agn_dec.pt")
                    torch.save(target_model.state_dict(), model_path / f"{seed}agn_target.pt")
                else:
                    pass
    else:
        if config['dataset'] in ('credit', 'iot'):
            return recon_record, losses
        elif config['dataset'] in ('mnist', 'cifar10', 'tiny'):
           
            return recon_record, psnr_record, ssim_record, losses
        else:
            return -1 
                
def evaluate_AGN_img(encoder, target_model, decoder, testloader, adv, target, config, device):
    encoder.eval()
    decoder.eval()
    target_model.eval()
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    target_mse = 0.0
    target_psnr = 0.0
    target_ssim = 0.0
    mse = nn.MSELoss()
    with torch.no_grad():
        for data, label in testloader:
            data[adv] = data[adv].to(device)
            data[target] = data[target].to(device)
            target_img = torch.cat((data[adv], data[target]), -1)
            
            enc_output = encoder(data[adv])

            target_output = target_model(data[target])


            target_recon = decoder(data[adv], enc_output, target_output)
           
            loss_target = mse(target_recon[:, :, :, config['dim']:], data[target])
            loss_psnr = psnr(target_recon[:, :, :, config['dim']:], data[target])
            loss_ssim = ssim(target_recon[:, :, :, config['dim']:], data[target])
        
            target_mse += loss_target.cpu().item()
            target_psnr += loss_psnr.cpu().item()
            target_ssim += loss_ssim.cpu().item()
            del enc_output, target_output

    print(f'AGN Test Mean: {target_mse/ len(testloader)}, AGN PSNR :{target_psnr/len(testloader)}, AGN SSIM:{target_ssim/len(testloader)}')
    return target_mse/ len(testloader), target_psnr/len(testloader),target_ssim/len(testloader)

def evaluate_AGN_table(encoder, target_model, decoder, testloader, adv, target, config, device):
    encoder.eval()
    decoder.eval()
    target_model.eval()
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    target_mse = 0.0
    target_psnr = 0.0
    target_ssim = 0.0
    mse = nn.MSELoss()
    with torch.no_grad():
        for data, label in testloader:
            data[adv] = data[adv].to(device)
            data[target] = data[target].to(device)
            target_img = torch.cat((data[adv], data[target]), -1)
            
            enc_output = encoder(data[adv])

            target_output = target_model(data[target])


            target_recon = decoder(data[adv], enc_output, target_output)
            
            loss_target = mse(target_recon[:, config['dim']:], data[target])
       
            target_mse += loss_target.cpu().item()
            
            del enc_output, target_output

    print(f'AGN Test Mean: {target_mse/ len(testloader)}, AGN PSNR :{target_psnr/len(testloader)}, AGN SSIM:{target_ssim/len(testloader)}')
    return target_mse/ len(testloader)
