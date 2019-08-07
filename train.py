import torch
import torch.nn as nn
from torch import optim
from torchvision import utils as vutils
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

import os
import time
import joblib
import argparse

from utils import inf_generator, calculate_psnr, calculate_ssim, rgb2ycbcr
from utils.lmdb_dataset import TrainDataset, ValDataset
from modules.models import ODEMSR

from google.cloud import storage
from sqlalchemy import create_engine

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def evaluate(model, update_step, opt, writer, bucket, engine):
    device = torch.device('cuda:0')
    model.eval()
    eval_paths = [os.path.join(opt['eval_path'], v) for v in ['Set14', 'Set5']]
    metrics_list = []

    for eval_path in eval_paths:
        eval_name = os.path.basename(eval_path)
        HQ_path = os.path.join(eval_path, eval_name) + '.lmdb'
        LQ_path = os.path.join(eval_path, eval_name) + '_LQ.lmdb'
        LQ_r_path = os.path.join(eval_path, eval_name) + '_LQ_restored.lmdb'

        eval_set = ValDataset(HQ_path, LQ_path, LQ_r_path)
        eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False,
                                 pin_memory=True, num_workers=0)

        psrn_rgb = 0.0
        psrn_y = 0.0
        ssim_rgb = 0.0
        ssim_y = 0.0

        for i, data_dict in enumerate(eval_loader):
            img_HQ = data_dict['img_GT']
            img_LQ = data_dict['img_LQ'].to(device)
            img_LQ_r = data_dict['img_LQ_r']

            with torch.no_grad():
                img_SR = model(img_LQ)
            if i == 0:
                imgs = torch.cat([img_HQ, img_SR.detach().cpu(), img_LQ_r], dim=0)
                grid = vutils.make_grid(imgs, nrow=3, normalize=False)
                tmp_image = T.ToPILImage()(grid)
                tmp_image.save('tmp_image.png')
                upload_to_cloud(bucket, 'tmp_image.png',
                                'odesr/image_progress/gen_step_{}'.
                                format(update_step * opt['update_freq']))
                if eval_name == 'Set5':
                    writer.add_image('Set5', grid, update_step)

            crop_size = opt['scale']
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
        INSERT into odemsr_val (set14_psnr_rgb, set14_psnr_y, set14_ssim_rgb,
        set14_ssim_y, set5_psnr_rgb, set5_psnr_y, set5_ssim_rgb,
        set5_ssim_y) values (%f, %f, %f, %f, %f, %f, %f, %f)
    ''' % tuple(metrics_list)
    engine.execute(query)
    model.train()


def upload_to_cloud(bucket, local_path, remote_path):
    blob = bucket.blob(remote_path)
    blob.upload_from_filename(filename=local_path)


def download_from_cloud(bucket, remote_path, local_path):
    blob = bucket.get_blob(remote_path)
    blob.download_to_filename(local_path)


def train(opt, resume=None):
    client = storage.Client()
    bucket = client.get_bucket('gloryofresearch')
    engine = create_engine(
        'mysql+pymysql://root:123@/model_monitoring?unix_socket=/cloudsql/helpful-quanta-248212:us-central1:model-monitoring-1')

    device = torch.device('cuda:0')

    netG = ODEMSR(opt['scale'], opt['hidden_size']).to(device)
    optimG = optim.Adam(netG.parameters(), lr=opt['lr'], betas=[0.9, 0.999])
    schedulerG = optim.lr_scheduler.StepLR(
        optimG, step_size=opt['lr_step_size'], gamma=0.5
        )
    pixel_loss = nn.L1Loss()
    print('Generator params: ', sum(p.numel() for p in netG.parameters()))
    writer = SummaryWriter('tensorboard/generator')

    start_step = 1
    if resume:
        print('Downloading ckeckpoints')
        download_from_cloud(bucket, resume, 'last_ckpt.pth')
        state_dicts = torch.load('last_ckpt.pth')
        netG.load_state_dict(state_dicts['netG'])
        optimG.load_state_dict(state_dicts['optimG'])
        schedulerG.load_state_dict(state_dicts['schedulerG'])

        start_step = state_dicts['gen_step']
        print('Resuming from step: {}'.format(start_step))

    train_set = TrainDataset(opt['train_HQ_path'], opt['train_LQ_path'])
    train_loader = DataLoader(train_set, batch_size=opt['batch_size'],
                              shuffle=True, num_workers=0)
    train_iterator = inf_generator(train_loader)

    start_time = time.time()
    for train_step in range(1, opt['train_steps']):
        netG.zero_grad()

        imgs_dict = next(train_iterator)
        img_HR = imgs_dict['img_GT'].to(device)
        img_LR = imgs_dict['img_LQ'].to(device)
        img_SR = netG(img_LR)

        nfe_f = netG.ode.nfe
        netG.ode.nfe = 0

        lossG = pixel_loss(img_SR, img_HR)
        lossG.backward()
        optimG.step()

        nfe_b = netG.ode.nfe
        netG.ode.nfe = 0

        schedulerG.step()
        if train_step % opt['update_freq'] == 0:
            elapsed_time = time.time() - start_time

            state_dict = {
                'netG': netG.state_dict(),
                'optimG': optimG.state_dict(),
                'schedulerG': schedulerG.state_dict(),
                'gen_step': train_step
            }
            torch.save(state_dict, 'tmp_checkpoint.pth'.format(train_step))
            upload_to_cloud(bucket, 'tmp_checkpoint.pth',
                            'odesr/model_checkpoints/gen_step_{}.pth'
                            .format(train_step))

            update_step = train_step // opt['update_freq']
            writer.add_scalar('Loss', lossG.item(), update_step)
            writer.add_scalar('NFE_F', nfe_f, update_step)
            writer.add_scalar('NFE_B', nfe_b, update_step)

            query = '''
                INSERT into odemsr_train (time, loss, nfe_f, nfe_b)
                VALUES ({}, {}, {}, {})
                '''.format(elapsed_time, lossG.item(), nfe_f, nfe_b)
            engine.execute(query)

            evaluate(netG, update_step, opt, writer, bucket, engine)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('opt', type=str, help='Options path')
    parser.add_argument('--resume', type=str, help='Model ckpt')
    args = parser.parse_args()

    opt = joblib.load(args.opt)

    opt['eval_path'] = '../val.lmdb'
    print(opt)
    train(opt, args.resume)
