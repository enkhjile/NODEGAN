import torch
import torch.nn as nn
from torch import optim
from torchvision import utils as vutils
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
# from tensorboardX import SummaryWriter

import os
import time
import argparse

from utils import inf_generator, calculate_psnr, calculate_ssim, rgb2ycbcr
from utils.lmdb_dataset import TrainDataset, ValDataset
from modules.models import ODESR
from modules import Normalize, UnNormalize, CosineAnnealingLR_Restart
from onecyclelr.onecyclelr import OneCycleLR
from torch.optim.lr_scheduler import ExponentialLR

from google.cloud import storage
from sqlalchemy import create_engine

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()

parser.add_argument('--train_HQ_path', default='/mnt/disks/ssd/train.lmdb/DF2K.lmdb')
parser.add_argument('--train_LQ_path', default='/mnt/disks/ssd/train.lmdb/DF2K_LQ.lmdb')
parser.add_argument('--eval_path', default='/mnt/disks/ssd/val.lmdb')
parser.add_argument('--scale', default=4)
parser.add_argument('--crop_size', default=128)
parser.add_argument('--train_steps', default=10000)
parser.add_argument('--lr', default=5e-4)
parser.add_argument('--batch_size', default=256)
parser.add_argument('--hidden_size', default=64)
parser.add_argument('--tol', default=1e-8)
parser.add_argument('--update_freq', default=50)
parser.add_argument('--resume', default=None)
parser.add_argument('--augment_dim', default=5)
# parser.add_argument('--train_table')

args = parser.parse_args()
print(args)


def upload_to_cloud(bucket, local_path, remote_path):
    blob = bucket.blob(remote_path)
    blob.upload_from_filename(filename=local_path)


def download_from_cloud(bucket, remote_path, local_path):
    blob = bucket.get_blob(remote_path)
    blob.download_to_filename(local_path)


def evaluate(model, update_step, writer, bucket, engine):
    device = torch.device('cuda:0')
    model.eval()
    eval_paths = [os.path.join(args.eval_path, v) for v in ['Set14', 'Set5']]
    metrics_list = []

    unnorm = UnNormalize(0.5, 0.5)
    for eval_path in eval_paths:
        eval_name = os.path.basename(eval_path)
        HQ_path = os.path.join(eval_path, eval_name) + '.lmdb'
        LQ_path = os.path.join(eval_path, eval_name) + '_LQ.lmdb'
        LQ_r_path = os.path.join(eval_path, eval_name) + '_LQ_restored.lmdb'

        eval_set = ValDataset(HQ_path, LQ_path, LQ_r_path, args.scale)
        eval_loader = DataLoader(
            eval_set, batch_size=1, shuffle=False, num_workers=4)

        psrn_rgb = 0.0
        psrn_y = 0.0
        ssim_rgb = 0.0
        ssim_y = 0.0

        for i, data_dict in enumerate(eval_loader):
            img_HQ = data_dict['img_GT']
            img_LQ = data_dict['img_LQ'].to(device)
            img_LQ_r = data_dict['img_LQ_r']

            with torch.no_grad():
                # SR image range [-1, 1]
                img_SR = model(img_LQ)
                # SR image range [0, 1]
                img_SR = unnorm(img_SR)
            if i == 0:
                imgs = torch.cat([img_HQ, img_SR.detach().cpu(), img_LQ_r], dim=0)
                grid = vutils.make_grid(imgs, nrow=3, normalize=False)
                tmp_image = T.ToPILImage()(grid)
                tmp_image.save('images/tmp_image.png')
                upload_to_cloud(bucket, 'images/tmp_image.png',
                                'odesr01_04/image_progress/{}/gen_step_{}'.
                                format(eval_name, update_step * args.update_freq))
                if eval_name == 'Set5':
                    writer.add_image('Set5', grid, update_step)

            crop_size = args.scale
            img_HQ_rgb = img_HQ[0].permute(2, 1, 0).cpu(). \
                numpy()[crop_size:-crop_size, crop_size:-crop_size, :]
            img_SR_rgb = img_SR[0].permute(2, 1, 0).detach().cpu(). \
                numpy()[crop_size:-crop_size, crop_size:-crop_size, :]
            img_HQ_y = rgb2ycbcr(img_HQ_rgb)
            img_SR_y = rgb2ycbcr(img_SR_rgb)

            psrn_rgb += calculate_psnr(img_HQ_rgb * 255, img_SR_rgb * 255)
            psrn_y += calculate_psnr(img_HQ_y * 255, img_SR_y * 255)
            ssim_rgb += calculate_ssim(img_HQ_rgb * 255, img_SR_rgb * 255)
            ssim_y += calculate_ssim(img_HQ_y * 255, img_SR_y * 255)

        psrn_rgb = psrn_rgb / len(eval_loader.dataset)
        psrn_y = psrn_y / len(eval_loader.dataset)
        ssim_rgb = ssim_rgb / len(eval_loader.dataset)
        ssim_y = ssim_y / len(eval_loader.dataset)

        metrics_list.extend([psrn_rgb, psrn_y, ssim_rgb, ssim_y])

        if eval_name == 'Set5':
            writer.add_scalar('psrn_rgb', psrn_rgb, update_step)
            writer.add_scalar('psrn_y', psrn_y, update_step)
            writer.add_scalar('ssim_rgb', ssim_rgb, update_step)
            writer.add_scalar('ssim_y', ssim_y, update_step)

    query = '''
        INSERT INTO odesr01_04_val
            (set14_psnr_rgb, set14_psnr_y, set14_ssim_rgb, set14_ssim_y,
            set5_psnr_rgb, set5_psnr_y, set5_ssim_rgb, set5_ssim_y)
        VALUES (%f, %f, %f, %f, %f, %f, %f, %f)
    ''' % tuple(metrics_list)
    engine.execute(query)
    model.train()


def train(resume=None):
    # connect to GS
    print('Connecting to engines..')
    client = storage.Client()
    bucket = client.get_bucket('gloryofresearch')
    # connect to MySql
    engine = create_engine(
        'mysql+pymysql://root:123@/model_monitoring?unix_socket=/cloudsql/helpful-quanta-248212:us-central1:model-monitoring-1')

    print('Done')
    device = torch.device('cuda:0')
    netG = ODESR(args.scale, args.hidden_size, device, args.tol).to(device)
    optimG = optim.AdamW(netG.parameters(), lr=args.lr, betas=[0.9, 0.999],
                         weight_decay=1e-4)
    schedulerG = OneCycleLR(optimG, num_steps=args.train_steps,
                            lr_range=(args.lr, args.lr*2))
    # schedulerG = ExponentialLR(optimG, gamma=5)

    pixel_loss = nn.MSELoss()
    query = 'create table if not exists odesr01_04_train like odesr01_02_train'
    engine.execute(query)
    query = 'create table if not exists odesr01_04_val like odesr01_02_val'
    engine.execute(query)

    print('Generator params: ', sum(p.numel() for p in netG.parameters()))
    now = datetime.now()
    date = now.strftime('%Y-%m-%d')
    writer = SummaryWriter('tensorboard/{}'.format(date))

    start_step = 1
    if resume:
        print('Downloading checkpoints')
        download_from_cloud(bucket, resume, 'model_ckpt/last_ckpt.pth')
        state_dicts = torch.load('model_ckpt/last_ckpt.pth')
        netG.load_state_dict(state_dicts['netG'])
        optimG.load_state_dict(state_dicts['optimG'])
        schedulerG.load_state_dict(state_dicts['schedulerG'])

        start_step = state_dicts['gen_step'] + 1
        print('Resuming from step: {}'.format(start_step))
        print('Learning rage: {}'.format(schedulerG.get_lr()))

    train_set = TrainDataset(
        args.train_HQ_path, args.train_LQ_path, args.crop_size, args.scale)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=8,
        pin_memory=False)
    train_iterator = inf_generator(train_loader)

    start_time = time.time()
    norm = Normalize(0.5, 0.5)
    print('Start training...')
    for train_step in range(start_step, args.train_steps + 1):
        netG.zero_grad()

        imgs_dict = next(train_iterator)

        img_HR = imgs_dict['img_GT'].to(device)
        # Normalize HR image to [-1, 1]
        img_HR = norm(img_HR)
        img_LR = imgs_dict['img_LQ'].to(device)
        # SR image range [-1, 1]
        img_SR = netG(img_LR)

        nfe_f = netG.ode.nfe
        netG.ode.nfe = 0

        lossG = pixel_loss(img_SR, img_HR)
        lossG.backward()
        optimG.step()

        nfe_b = netG.ode.nfe
        netG.ode.nfe = 0

        schedulerG.step()
        if train_step % args.update_freq == 0:
            elapsed_time = time.time() - start_time
            state_dict = {
                'netG': netG.state_dict(),
                'optimG': optimG.state_dict(),
                'schedulerG': schedulerG.state_dict(),
                'gen_step': train_step
            }
            torch.save(
                state_dict, 'model_ckpt/tmp_checkpoint.pth'.format(train_step))
            upload_to_cloud(bucket, 'model_ckpt/tmp_checkpoint.pth',
                            'odesr01_04/model_checkpoints/gen_step_{}.pth'
                            .format(train_step))

            update_step = train_step // args.update_freq
            writer.add_scalar('Loss', lossG.item(), train_step)
            writer.add_scalar('NFE_F', nfe_f, update_step)
            writer.add_scalar('NFE_B', nfe_b, update_step)

            query = '''
                INSERT INTO odesr01_04_train
                    (time, loss, nfe_f, nfe_b)
                VALUES ({}, {}, {}, {})
                '''.format(elapsed_time, lossG.item(), nfe_f, nfe_b)
            engine.execute(query)

            evaluate(netG, update_step, writer, bucket, engine)
            start_time = time.time()


if __name__ == "__main__":
    if not os.path.exists('model_ckpt'):
        os.makedirs('model_ckpt')
    if not os.path.exists('images'):
        os.makedirs('images')
    train(args.resume)
