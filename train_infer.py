import time
import h5py
import gc
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import models.utils as utils
from cine_dataset import CineDataset_MC
import matplotlib.pyplot as plt
from metrics import calmetric, calmetric_new
from warmup_scheduler import GradualWarmupScheduler
from models.CRUNet_MR import CRUNet_D_NWS
import os
import random 
import numpy as np
import argparse
import wandb
from losses import SSIMLoss
from tqdm import tqdm

os.environ["WANDB__SERVICE_WAIT"] = "300"

models_name = {
    "CRUNet_MR": CRUNet_D_NWS,
}

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
setup_seed(0)

def clip_images(image):
    for i in range(image.shape[0]):
        img = np.abs(image[i])
        lower_bound = np.percentile(img, 1)
        upper_bound = np.percentile(img, 99)
        clipped_image = np.clip(img, lower_bound, upper_bound)
        image[i] = (clipped_image - lower_bound) / (upper_bound - lower_bound)
    return image

def visualize(img, path):
    img = clip_images(img)
    fig, axs = plt.subplots(3,4,figsize=(10,10))
    plt.tight_layout()
    for i in range(3):
        for j in range(4):
            axs[i,j].imshow(np.abs(img[i*4+j]), cmap="gray")
            axs[i,j].set_title(f"frame {i*4+j}")
            axs[i,j].axis('off')
    plt.savefig(path)
    plt.close()
    
    
def sens_expand(x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    """
    Forward operator: from coil-combined image-space to k-space.
    """
    return utils.fft2c(utils.complex_mul(x, sens_maps))

def sens_reduce(x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    """
    Backward operator: from k-space to coil-combined image-space.
    """
    x = utils.ifft2c(x)
    return utils.complex_mul(x, utils.complex_conj(sens_maps)).sum(
        dim=2, keepdim=True,
    )

def build_loader(args, folder_path):
    folder_path = os.path.join(folder_path, args.axis)
    files = os.listdir(folder_path)
    dataset = CineDataset_MC(files = files, folder_path=folder_path, mode=args.mode)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def check_path(folder_path, model_name, mode, axis, tv, fname):
    model_dir = os.path.join(folder_path, model_name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    mode_dir = os.path.join(model_dir, mode)
    if not os.path.exists(mode_dir):
        os.mkdir(mode_dir)

    axis_dir = os.path.join(mode_dir, axis)
    if not os.path.exists(axis_dir):
        os.mkdir(axis_dir)
        
    if tv is None:
        tv_dir = axis_dir
    else:
        tv_dir = os.path.join(axis_dir, tv)
        if not os.path.exists(tv_dir):
            os.mkdir(tv_dir)
    
    if fname is not None:
        if fname=="gnd" or fname=="und":
            file_dir = os.path.join(tv_dir, fname)
        else:
            file_dir = os.path.join(tv_dir, f"rec_{fname}")
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        return file_dir
    else:
        return tv_dir


def val_test(model, data_loader, mode="val", philips=True):
    model.eval()
    test_loss = 0
    nmses = []
    psnrs = []
    ssims = []
    distses = []
    haars = []
    names = []
    img_unds = []
    img_recs = []
    img_gnds = []
    criterion = torch.nn.MSELoss().cuda()
    if not philips:
        for i, (full_kspace, und_kspace, mask, sense_map, name) in enumerate(data_loader):
            sense_map = torch.view_as_real(sense_map).float().cuda()
            full_image = torch.view_as_complex(sens_reduce(torch.view_as_real(full_kspace).cuda(), sense_map).squeeze(2)).cuda()
            und_image = torch.view_as_complex(sens_reduce(torch.view_as_real(und_kspace).cuda(), sense_map).squeeze(2)).cuda()
            with torch.no_grad():
                rec_image = model(und_kspace.cuda(), mask.float().cuda(), sense_map)

            gnd_image = torch.view_as_real(full_image)
            t_loss = criterion(rec_image, gnd_image)
            test_loss+=t_loss.item()
            names.append(name[0])
            
            rec_img = torch.view_as_complex(rec_image).squeeze(0).cpu().numpy()
            gnd_img = torch.view_as_complex(gnd_image).squeeze(0).cpu().numpy()
            und_img = und_image.squeeze(0).cpu().numpy()
            img_unds.append(und_img)
            img_recs.append(rec_img)
            img_gnds.append(gnd_img)
            if mode=="val":
                [psnr_array, ssim_array, nmse_array] = calmetric(np.abs(rec_img).transpose(1,2,0), np.abs(gnd_img).transpose(1,2,0))
                nmses.append(np.mean(nmse_array))
                psnrs.append(np.mean(psnr_array))
                ssims.append(np.mean(ssim_array))
            else:
                [psnr_array, ssim_array, nmse_array, dists_array, haar_array] = calmetric_new(np.abs(rec_img).transpose(1,2,0), np.abs(gnd_img).transpose(1,2,0))
                psnrs.append(np.mean(psnr_array))
                ssims.append(np.mean(ssim_array))
                nmses.append(np.mean(nmse_array))
                distses.append(np.mean(dists_array))
                haars.append(np.mean(haar_array))
        if mode=="val":
            return float(test_loss/(i+1)), nmses, psnrs, ssims, names, img_unds, img_recs, img_gnds
        else:
            return float(test_loss/(i+1)), nmses, psnrs, ssims, distses, haars, names, img_unds, img_recs, img_gnds
    else:
        for i, (und_kspace, mask, sense_map, name) in enumerate(data_loader):
            sense_map = torch.view_as_real(sense_map).float().cuda()
            und_image = torch.view_as_complex(sens_reduce(torch.view_as_real(und_kspace).cuda(), sense_map).squeeze(2)).cuda()
            with torch.no_grad():
                rec_image = model(und_kspace.cuda(), mask.float().cuda(), sense_map)
            names.append(name[0])
            rec_img = torch.view_as_complex(rec_image).squeeze(0).cpu().numpy()
            und_img = und_image.squeeze(0).cpu().numpy()
            img_unds.append(clip_images(und_img))
            img_recs.append(clip_images(rec_img))
        return names, img_unds, img_recs

def process_val_test(args, model, data_loader, f_name, epoch, best_psnr, best_ssim, mode="val", philips=True):
    if args.save_pic_path=="":
        os.makedirs("./pic_save", exist_ok=True)
        args.save_pic_path = "./pic_save"
    if args.save_val_path=="":
        os.makedirs("./val_save", exist_ok=True)
        args.save_val_path = "./val_save"
    if args.save_weight_path=="":
        os.makedirs("./weight_save", exist_ok=True)
        args.save_weight_path = "./weight_save"
        
    if mode=="val":
        test_loss, nmses, psnrs, ssims, names, img_unds, img_recs, img_gnds = val_test(model, data_loader, philips=philips)
        c_psnr = np.mean(psnrs)
        c_ssim = np.mean(ssims)
        c_nmse = np.mean(nmses)
        wandb.log({"Test_Loss": float(test_loss), 
                    "Val_NMSE": c_nmse, 
                    "Val_PSNR": c_psnr,
                    "Val_SSIM": c_ssim})
        
        if c_ssim>best_ssim:
            best_ssim = c_ssim
            weight_name = f"{f_name}_best.pth"
            weight_rec_path = check_path(args.save_weight_path, args.model_name, args.mode, args.axis, None, None)
            torch.save(model.state_dict(), os.path.join(weight_rec_path, weight_name))
            print('model parameters saved at %s' % os.path.join(weight_rec_path, weight_name))
            
            val_path = check_path(args.save_val_path, args.model_name, args.mode, args.axis, mode, f_name)
            for i in range(len(names)):
                with h5py.File(os.path.join(val_path, f"{names[i]}.h5"), 'w') as f:
                    f["und"] = img_unds[i]
                    f["rec"] = img_recs[i]
                    f["gnd"] = img_gnds[i]
                    f["mean_nmse"] = c_nmse
                    f["mean_psnr"] = c_psnr
                    f["mean_ssim"] = c_ssim
            print("val psnr:", c_psnr)
            print("val ssim:", c_ssim)
            print("val nmse:", c_nmse)

            und_path = check_path(args.save_pic_path, args.model_name, args.mode, args.axis, mode, "und")
            gnd_path = check_path(args.save_pic_path, args.model_name, args.mode, args.axis, mode, "gnd")
            rec_path = check_path(args.save_pic_path, args.model_name, args.mode, args.axis, mode, f_name)
            for i in range(len(names)):
                if epoch==0:
                    visualize(img_unds[i], os.path.join(und_path, f"{names[i]}.png"))
                    visualize(img_gnds[i], os.path.join(gnd_path, f"{names[i]}.png"))
                visualize(img_recs[i], os.path.join(rec_path, f"{names[i]}.png"))
    elif mode=="test":
        if not philips:
            test_loss, nmses, psnrs, ssims, distses, haars, names, img_unds, img_recs, img_gnds = val_test(model, data_loader, mode, philips=philips)
            val_path = check_path(args.save_val_path, args.model_name, args.mode, args.axis, mode, f_name)
            for i in range(len(names)):
                with h5py.File(os.path.join(val_path, f"{names[i]}.h5"), 'w') as f:
                    f["und"] = img_unds[i]
                    f["rec"] = img_recs[i]
                    f["gnd"] = img_gnds[i]
                    f["mean_nmse"] = np.mean(nmses)
                    f["mean_psnr"] = np.mean(psnrs)
                    f["mean_ssim"] = np.mean(ssims)
                    f["mean_dists"] = np.mean(distses)
                    f["mean_haarpsi"] = np.mean(haars)
                    f["std_nmse"] = np.std(nmses)
                    f["std_psnr"] = np.std(psnrs)
                    f["std_ssim"] = np.std(ssims)
                    f["std_dists"] = np.std(distses)
                    f["std_haarpsi"] = np.std(haars)
            print("test psnr:", np.mean(psnrs))
            print("test ssim:", np.mean(ssims))
            print("test dists:", np.mean(distses))
            print("test haars:", np.mean(haars))
        else:
            names, img_unds, img_recs = val_test(model, data_loader, mode, philips=philips)
            val_path = check_path(args.save_val_path, args.model_name, args.mode, args.axis, mode, f_name)
            for i in range(len(names)):
                with h5py.File(os.path.join(val_path, f"{names[i]}.h5"), 'w') as f:
                    f["und"] = img_unds[i]
                    f["rec"] = img_recs[i]
    return best_psnr, best_ssim


def train_infer(args, train=True, infer_weight_path=None):
    best_psnr = 0
    best_ssim = 0
    lr = args.lr
    num_epoch = args.num_epoch
    weight_decay = args.weight_decay
    warmup_epoch = args.warmup_episodes
    interval = args.interval
    f_name = args.model_name
        
    model = models_name[args.model_name]()
    model = model.cuda()
        
    if train:
        mode="08"
        if args.mode=="AccFactor04":
            mode = "04"
        elif args.mode=="AccFactor10":
            mode = "10"
        elif args.mode=="AccFactor16":
            mode = "16"
    
        criterion = torch.nn.MSELoss().cuda()
        criterion0 = torch.nn.L1Loss().cuda()
        criterion1 = SSIMLoss().cuda()
        optimizer = optim.AdamW(model.parameters(), lr=lr,betas=(0.9,0.999),eps=1e-8, weight_decay=weight_decay)
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch-warmup_epoch, eta_min=1e-4)  # eta_min from 1e-5 to 1e-6
        scheduler = GradualWarmupScheduler(optimizer,
                multiplier=1,total_epoch=warmup_epoch,
                after_scheduler=scheduler_cosine)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Total trainable params: %d' % pytorch_total_params)
        g_loss = 1e4
        
        if args.axis=="lax":
            wan_name = f"CMR_{mode}_LAX_new"
        else:
            wan_name = f"CMR_{mode}_SAX_new"
        
        wandb.init(project=wan_name, name=f"{f_name}_{args.axis}_cascade5")  
        wandb.watch(model)
    
        train_loader = build_loader(args, args.train_path)
        val_loader = build_loader(args, args.val_path)
        test_loader = build_loader(args, args.test_path)
 
    
    if train:
        for epoch in range(0, num_epoch+1):
            gc.collect()
            torch.cuda.empty_cache()
            t_start = time.time()
            if epoch == num_epoch:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-5
            train_err = 0
            train_batches = 0
            model.train()
            pbar = tqdm(train_loader)
            for i, (full_kspace, und_kspace, mask, sense_map, _) in enumerate(pbar):
                sense_map = torch.view_as_real(sense_map).float().cuda()
                full_image = torch.view_as_complex(sens_reduce(torch.view_as_real(full_kspace).cuda(), sense_map).squeeze(2)).cuda()
                rec_image = model(und_kspace.cuda(), mask.float().cuda(), sense_map).float()
                gnd_image = torch.view_as_real(full_image).float().cuda()
                gnd_kspace = utils.fft2c(gnd_image).float().cuda()
                rec_ksapce = utils.fft2c(rec_image).float().cuda()

                loss = 0.25*criterion(rec_ksapce, gnd_kspace)+0.5*(criterion(rec_image, gnd_image)+criterion0(rec_image, gnd_image))+0.5*(criterion1(rec_image[...,0], gnd_image[...,0])+criterion1(rec_image[...,1], gnd_image[...,1]))
                
                train_err += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_batches += 1
                pbar.set_description(f"Epoch {epoch}, loss: {loss.item():.4f}")
                
            scheduler.step()
            t_end = time.time()
            train_err /= train_batches
            print(" Epoch {}/{}".format(epoch + 1, num_epoch))
            print(" time: {}s".format(t_end - t_start))
            print(" training loss:\t\t{:.6f}".format(train_err))
            wandb.log({"Train_Loss": train_err})

            if train_err<g_loss:
                weight_name = f"{f_name}_latest.pth"
                weight_rec_path = check_path(args.save_weight_path, args.model_name, args.mode, args.axis, None, None)
                torch.save(model.state_dict(), os.path.join(weight_rec_path, weight_name))
                print('model parameters saved at %s' % os.path.join(weight_rec_path, weight_name))
                g_loss = train_err
            
            if epoch%interval==0:
                gc.collect()
                torch.cuda.empty_cache()
                best_psnr, best_ssim = process_val_test(args, model, val_loader, f_name, epoch, best_psnr, best_ssim)
                if epoch==0:
                    _, _ = process_val_test(args, model, test_loader, f_name, epoch, best_psnr, best_ssim, mode="test")

        
        weight_name = f"{f_name}_latest.pth"
        weight_rec_path = check_path(args.save_weight_path, args.model_name, args.mode, args.axis, None, None)
        weight_path = os.path.join(weight_rec_path, weight_name)
        checkpoint = torch.load(weight_path)
        model.load_state_dict(checkpoint)
        _, _ = process_val_test(args, model, test_loader, f_name, epoch, best_psnr, best_ssim, mode="test")
    else:
        epoch = 0
        checkpoint = torch.load(infer_weight_path)
        model.load_state_dict(checkpoint)
        _, _ = process_val_test(args, model, test_loader, f_name, epoch, best_psnr, best_ssim, mode="test")    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="CRUNet_MR", help='model name')
    
    parser.add_argument('--train_path', type=str, default="", help='save path for training data') # needed for training
    parser.add_argument('--val_path', type=str, default="", help='save path for validation data') # needed for validation
    parser.add_argument('--test_path', type=str, default="", help='save path for testing data')  # needed for testing: hdf5 file path (e.g "./cine_Aov_cs.hdf5")
    parser.add_argument('--save_pic_path', type=str, default="", help='save path for images')
    parser.add_argument('--save_val_path', type=str, default="", help='save path for values')
    parser.add_argument('--save_weight_path', type=str, default="", help='save path for weights')
    
    parser.add_argument('--num_epoch', metavar='int', nargs=1, type=int, default=144, help='number of epochs')
    parser.add_argument('--batch_size', metavar='int', nargs=1, type=int, default=1, help='batch size')
    parser.add_argument('--lr', metavar='float', nargs=1, type=float, default=3e-4, help='initial learning rate')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--interval', type=int, default=12)
    parser.add_argument('--weight_decay', type=float, default=1e-3)  # from 1e-5 to 5e-5
    parser.add_argument('--warmup_episodes', type=int, default=10) # 25
    parser.add_argument('--axis', type=str, default="lax")
    parser.add_argument('--mode', type=str, default="AccFactor08")
    parser.add_argument('--pretrain', type=bool, default=False)
    args = parser.parse_args()
    
    infer_weight_path = ""  # ./weights/sax/Acc8_5/CRUNet_D_UP_latest.pth
    train_infer(args, False, infer_weight_path)
    
