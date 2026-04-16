
"""
Created on April 2019
Deep learning reconstruction library


@author: Abi Mehranian
abolfazl.mehranian@kcl.ac.uk

"""
import torch
import torch.nn as nn
from models.deeplib import zeroNanInfs, crop, uncrop
import numpy as np


class WeightedMSELoss(nn.Module):
    """MSE with higher weight on foreground (non-background) voxels.

    Voxels whose target value exceeds ``threshold_fraction * target.max()``
    are given ``foreground_weight`` × more gradient signal.  This prevents the
    loss from being dominated by the large number of zero-valued background
    voxels, giving lesions and brain tissue a stronger learning signal.
    """
    def __init__(self, foreground_weight=10.0, threshold_fraction=0.05):
        super().__init__()
        self.foreground_weight = foreground_weight
        self.threshold_fraction = threshold_fraction

    def forward(self, pred, target):
        threshold = target.detach().max() * self.threshold_fraction
        weight = torch.where(
            target > threshold,
            torch.full_like(target, self.foreground_weight),
            torch.ones_like(target),
        )
        return (weight * (pred - target).pow(2)).mean()


def _compute_ssim_psnr(pred_t, target_t):
    """Return mean SSIM and PSNR over a batch of reconstructed images.

    Parameters
    ----------
    pred_t, target_t : torch.Tensor, shape (B, 1, H, W)

    Returns
    -------
    ssim_val : float or None  — None when skimage is not installed
    psnr_val : float
    """
    pred   = pred_t.detach().cpu().numpy().squeeze(1).astype('float32')   # (B, H, W)
    target = target_t.detach().cpu().numpy().squeeze(1).astype('float32')

    mse     = np.mean((pred - target) ** 2, axis=(1, 2))
    max_val = target.max(axis=(1, 2))
    psnr_val = float(np.mean(20.0 * np.log10(max_val / (np.sqrt(mse) + 1e-8))))

    ssim_val = None
    try:
        from skimage.metrics import structural_similarity as _ssim
        ssim_val = float(np.mean([
            _ssim(pred[i], target[i], data_range=float(target[i].max()))
            for i in range(pred.shape[0])
        ]))
    except ImportError:
        pass

    return ssim_val, psnr_val


class ResUnit_v2(nn.Module):
    def __init__(self, depth, num_kernels, kernel_size,in_channels,is3d):
        super(ResUnit_v2, self).__init__()
        self.in_channels =in_channels
        self.relu = nn.ReLU(inplace=True)
        layers = []
        if is3d:
             layers.append(nn.Conv3d(in_channels, num_kernels, kernel_size, padding=1))
             layers.append(nn.BatchNorm3d(num_kernels))
             layers.append(nn.ReLU(inplace=True))
             for _ in range(depth-2):
                  layers.append(nn.Conv3d(num_kernels, num_kernels, kernel_size, padding=1))
                  layers.append(nn.BatchNorm3d(num_kernels))
                  layers.append(nn.ReLU(inplace=True))
             layers.append(nn.Conv3d(num_kernels, 1, kernel_size, padding=1))
             layers.append(nn.BatchNorm3d(1))             
        else:
             layers.append(nn.Conv2d(in_channels, num_kernels, kernel_size, padding=1))
             layers.append(nn.BatchNorm2d(num_kernels))
             layers.append(nn.ReLU(inplace=True))
             for _ in range(depth-2):
                  layers.append(nn.Conv2d(num_kernels, num_kernels, kernel_size, padding=1))
                  layers.append(nn.BatchNorm2d(num_kernels))
                  layers.append(nn.ReLU(inplace=True))
             layers.append(nn.Conv2d(num_kernels, 1, kernel_size, padding=1))
             layers.append(nn.BatchNorm2d(1))
        self.dcnn = nn.Sequential(*layers)

    def forward(self, x, y=None):
        identity = x
        if y is not None:
            x = torch.cat((x,y),dim=1)
        out = self.dcnn(x)
        out += identity
        out = self.relu(out)
        return out
   
class FBSEMnet_v3(nn.Module):
    def __init__(self, depth, num_kernels, kernel_size, in_channels=1, is3d=False, reg_cnn_model='resUnit'):
        super(FBSEMnet_v3, self).__init__()
        if reg_cnn_model.lower() == 'resunit':
            self.regularize = ResUnit_v2(depth, num_kernels, kernel_size, in_channels, is3d)
        else:
            raise ValueError(f"Unknown reg_cnn_model: {reg_cnn_model!r}. Only 'resUnit' is supported.")
        self.gamma = nn.Parameter(torch.rand(1),requires_grad=True)
        self.is3d = is3d
        
    def forward(self,PET,prompts,img=None,RS=None, AN=None, iSensImg = None, mrImg=None, niters = 10, nsubs=1, tof=False, psf=0,device ='cuda', crop_factor = 0):
         # e.g. crop_factor = 0.667
         
         batch_size = prompts.shape[0]
         device = torch.device(device)
         matrixSize = PET.image.matrixSize
         if 0<crop_factor<1: 
             Crop    = lambda x: crop(x,crop_factor,is3d=self.is3d)
             unCrop  = lambda x: uncrop(x,matrixSize[0],is3d=self.is3d)
         else:
             Crop    = lambda x: x
             unCrop  = lambda x: x         
         if self.is3d:  
             toTorch = lambda x: zeroNanInfs(Crop(torch.from_numpy(x).unsqueeze(1).to(device=device, dtype=torch.float32)))
             toNumpy = lambda x: zeroNanInfs(unCrop(x)).detach().cpu().numpy().squeeze(1).astype('float32')
             if iSensImg is None:
                  iSensImg = PET.iSensImageBatch3D(AN, nsubs, psf).astype('float32') 
             if img is None:
                  img =  np.ones([batch_size,matrixSize[0],matrixSize[1],matrixSize[2]],dtype='float32')
             if batch_size ==1:
                 if iSensImg.ndim==4:  iSensImg = iSensImg[None].astype('float32') 
                 if img.ndim==3:  img = img[None].astype('float32')
         else:
             reShape = lambda x: x.reshape([batch_size,matrixSize[0],matrixSize[1]],order='F')
             Flatten = lambda x: x.reshape([batch_size,matrixSize[0]*matrixSize[1]],order='F')
             toTorch = lambda x: zeroNanInfs(Crop(torch.from_numpy(reShape(x)).unsqueeze(1).to(device=device, dtype=torch.float)))
             toNumpy = lambda x: zeroNanInfs(Flatten((unCrop(x)).detach().cpu().numpy().squeeze(1)))
             if iSensImg is None:
                  iSensImg = PET.iSensImageBatch2D(AN, nsubs, psf) 
             if img is None:
                  img =  np.ones([batch_size,matrixSize[0]*matrixSize[1]],dtype='float32')  
             if batch_size ==1:
                 if iSensImg.ndim==2:  iSensImg = iSensImg[None].astype('float32') 
                 if img.ndim==1:  img = img[None].astype('float32')
         if mrImg is not None:
              mrImg = Crop(mrImg)
         imgt = toTorch(img)

         for i in range(niters):
              for s in range(nsubs):
                   if self.is3d:
                        img_em = img*PET.forwardDivideBackwardBatch3D(img, prompts, RS, AN, nsubs, s, psf)*iSensImg[:,s,:,:,:]
                        img_emt = toTorch(img_em) 
                        img_regt = zeroNanInfs(self.regularize(imgt,mrImg)) 
                        S = toTorch(iSensImg[:,s,:,:,:])
                   else:
                        img_em = img*PET.forwardDivideBackwardBatch2D(img, prompts, RS, AN, nsubs, s, tof, psf)*iSensImg[:,s,:]
                        img_emt = toTorch(img_em) 
                        img_regt = zeroNanInfs(self.regularize(imgt,mrImg)) 
                        S = toTorch(iSensImg[:,s,:])
                   imgt = 2*img_emt/((1 - self.gamma*S*img_regt) + torch.sqrt((1 - self.gamma*S*img_regt)**2 + 4*self.gamma*S*img_emt)) 
                   img = toNumpy(imgt)
                   del img_em, img_emt, img_regt, S
         del iSensImg, prompts, RS, AN, PET, img

         return unCrop(imgt)

def Trainer(PET, model, opts, train_loader, valid_loader=None):
    from models.deeplib import dotstruct, setOptions, imShowBatch,crop
    import torch.optim as optim
    import os
    
    g = dotstruct()
    g.psf_cm = 0.15
    g.niters = 10
    g.nsubs = 6
    g.lr = 0.001
    g.epochs = 100
    g.in_channels = 1
    g.save_dir = os.getcwd()
    g.model_name = 'fbsem-pm-01'
    g.display = True
    g.disp_figsize=(20,10)
    g.save_from_epoch = None
    g.crop_factor = 0.3
    g.do_validation = True
    g.device = 'cpu'
    g.mr_scale = 5
    g.loss_type = 'mse'          # 'mse' or 'weighted_mse'
    g.foreground_weight = 10.0   # weight multiplier for foreground voxels (weighted_mse only)
    # LR scheduling (ReduceLROnPlateau)
    g.lr_scheduler = 'plateau'   # 'plateau' or None to disable
    g.lr_patience = 5            # epochs without improvement before LR reduction
    g.lr_factor = 0.5            # multiply LR by this factor on plateau
    g.lr_min = 1e-6              # floor for LR reduction
    # Early stopping
    g.early_stop_patience = 15   # epochs without improvement before stopping (None to disable)

    g = setOptions(g, opts)

    if not os.path.exists(g.save_dir):
        os.makedirs(g.save_dir)

    if g.loss_type == 'weighted_mse':
        loss_fn = WeightedMSELoss(g.foreground_weight)
    else:
        loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=g.lr)
    if g.lr_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=g.lr_factor,
            patience=g.lr_patience, min_lr=g.lr_min, verbose=True,
        )
    else:
        scheduler = None

    toNumpy = lambda x: x.detach().cpu().numpy().astype('float32')

    train_losses = []
    valid_losses = []
    valid_ssims  = []
    valid_psnrs  = []
    gamma = []
    best_valid_loss  = float('inf')
    early_stop_count = 0
    
    for e in range(g.epochs):
         
         running_loss = 0
         for sinoLD, imgHD, AN, _,_, _, mrImg, _, _,index in train_loader: 
             #torch.cuda.empty_cache()
             AN=toNumpy(AN)
             RS = None
             sinoLD=toNumpy(sinoLD)
             imgHD = imgHD.to(g.device,dtype=torch.float32).unsqueeze(1)
    
             if g.in_channels==2:
                  mrImg = g.mr_scale*mrImg/mrImg.max()
                  mrImg = mrImg.to(g.device,dtype=torch.float32).unsqueeze(1)
             else:
                  mrImg = None
             optimizer.zero_grad()
             img = model.forward(PET,prompts = sinoLD,AN=AN, mrImg = mrImg,\
                                 niters=g.niters, nsubs = g.nsubs, psf=g.psf_cm, device=g.device, crop_factor=g.crop_factor)#, 
             loss = loss_fn(img,imgHD)
             loss.backward()
             optimizer.step()
             running_loss+=loss.item()
             
             if torch.isnan(model.gamma) or model.gamma.data<0:
                 model.gamma.data = torch.Tensor([0.01]).to(g.device,dtype=torch.float32)
             if g.display:
                 imShowBatch(crop(toNumpy(img).squeeze(),0.3), figsize = g.disp_figsize)
                 gam = model.gamma.clone().detach().cpu().numpy()[0]
                 print(f"gamma: {gam}")
                 gamma.append(gam)
             del sinoLD, AN, RS, mrImg, index
    
         else:
             train_losses.append(running_loss/len(train_loader))
             print(f"Epoch: {e+1}/{g.epochs}, Training loss: {train_losses[e]:.3f}")
             if g.do_validation:
                 valid_loss   = 0.0
                 ssim_sum     = 0.0
                 psnr_sum     = 0.0
                 ssim_tracked = False
                 with torch.no_grad():
                     model.eval()
                     for sinoLD, imgHD, AN, _, _, _, mrImg, _, _, index in valid_loader:
                         AN     = toNumpy(AN)
                         sinoLD = toNumpy(sinoLD)
                         imgHD  = imgHD.to(g.device, dtype=torch.float32).unsqueeze(1)
                         if g.in_channels == 2:
                             mrImg = g.mr_scale * mrImg / mrImg.max()
                             mrImg = mrImg.to(g.device, dtype=torch.float32).unsqueeze(1)
                         else:
                             mrImg = None
                         img = model.forward(PET, prompts=sinoLD, AN=AN, mrImg=mrImg,
                                             niters=g.niters, nsubs=g.nsubs, psf=g.psf_cm,
                                             device=g.device, crop_factor=g.crop_factor)
                         valid_loss += loss_fn(img, imgHD).item()
                         ssim_val, psnr_val = _compute_ssim_psnr(img, imgHD)
                         if ssim_val is not None:
                             ssim_sum += ssim_val
                             ssim_tracked = True
                         psnr_sum += psnr_val
                 n_val = len(valid_loader)
                 valid_losses.append(valid_loss / n_val)
                 valid_psnrs.append(psnr_sum / n_val)
                 if ssim_tracked:
                     valid_ssims.append(ssim_sum / n_val)
                 model.train()
                 ssim_str = f", SSIM: {valid_ssims[-1]:.4f}" if ssim_tracked else ""
                 print(f"Epoch: {e+1}/{g.epochs}, "
                       f"Val loss: {valid_losses[-1]:.4f}, "
                       f"PSNR: {valid_psnrs[-1]:.2f} dB{ssim_str}")
             # ── LR scheduler step ──────────────────────────────────────────
             monitor_loss = valid_losses[-1] if (g.do_validation and valid_losses) else train_losses[-1]
             if scheduler is not None:
                 scheduler.step(monitor_loss)

             # ── Early stopping ─────────────────────────────────────────────
             stop_early = False
             if g.early_stop_patience is not None and g.do_validation and valid_losses:
                 if monitor_loss < best_valid_loss:
                     best_valid_loss  = monitor_loss
                     early_stop_count = 0
                 else:
                     early_stop_count += 1
                     if early_stop_count >= g.early_stop_patience:
                         print(f"Early stopping at epoch {e+1} "
                               f"(no improvement for {g.early_stop_patience} epochs).")
                         stop_early = True

             # ── Checkpoint ────────────────────────────────────────────────
             is_last = stop_early or e == (g.epochs - 1)
             if ((g.save_from_epoch is not None) and (g.save_from_epoch <= e)) or is_last:
                  g.state_dict   = model.state_dict()
                  g.train_losses = train_losses
                  g.valid_losses = valid_losses
                  g.valid_psnrs  = valid_psnrs
                  g.valid_ssims  = valid_ssims
                  g.training_idx = train_loader.sampler.indices
                  g.gamma        = gamma

                  checkpoint = g.as_dict()
                  save_path  = os.path.join(g.save_dir, f"{g.model_name}-epo-{e}.pth")
                  torch.save(checkpoint, save_path)
                  print(f"Saved checkpoint: {save_path}")

             torch.cuda.empty_cache()
             if stop_early:
                 break

    return {
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'valid_psnrs':  valid_psnrs,
        'valid_ssims':  valid_ssims,
        'gamma':        gamma,
    }
             
             
def fbsemInference(dl_model_flname, PET, sinoLD, AN, mrImg, niters=None, nsubs = None, device='cpu'):

    toNumpy = lambda x: x.detach().cpu().numpy().astype('float32')

    g = torch.load(dl_model_flname, map_location=torch.device(device))
    
    reg_cnn_model = g.get('reg_cnn_model', g.get('reg_ccn_model', 'resUnit'))
    model = FBSEMnet_v3(g['depth'], g['num_kernels'], g['kernel_size'],
                        g['in_channels'], g['is3d'], reg_cnn_model).to(device)
    model.load_state_dict(g['state_dict'])
    
    AN=toNumpy(AN)
    RS = None
    sinoLD = toNumpy(sinoLD)

    if g['in_channels']==2:
         mrImg = g['mr_scale']*mrImg/mrImg.max()
         mrImg = mrImg.to(device,dtype=torch.float32).unsqueeze(1)
    else:
         mrImg = None
    niters = niters or g['niters']
    nsubs = nsubs or g['nsubs']
        
    with torch.no_grad():
        model.eval()
        img = model.forward(PET,prompts = sinoLD,AN=AN, mrImg = mrImg,\
                        niters=niters, nsubs = nsubs, psf=g['psf_cm'], device=device, crop_factor=g['crop_factor'])#, 
    return toNumpy(img).squeeze()