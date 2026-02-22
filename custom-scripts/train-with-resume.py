import torch
import os
import sys
sys.path.append(os.getcwd())
from models.autoencoder import AutoEncoder
import time
import argparse
from datetime import datetime
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models.utils import AverageMeter, str2bool
from dataset.dataset import CompressDataset
from args.shapenet_args import parse_shapenet_args
from args.semantickitti_args import parse_semantickitti_args
from torch.optim.lr_scheduler import StepLR
from models.Chamfer3D.dist_chamfer_3D import chamfer_3DDist
chamfer_dist = chamfer_3DDist()


def train(args):
    start = time.time()

    if args.batch_size > 1:
        print('The performance will degrade if batch_size is larger than 1!')

    if args.compress_normal:
        args.in_fdim = 6

    # ------------------------
    # LOAD DATASETS
    # ------------------------
    print("[DEBUG] Loading training dataset...")
    t0 = time.time()
    train_dataset = CompressDataset(
        data_path=args.train_data_path,
        cube_size=args.train_cube_size,
        batch_size=args.batch_size
    )
    print(f"[DEBUG] Training dataset initialized in {time.time() - t0:.2f}s")

    print("[DEBUG] Creating train DataLoader...")
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=args.batch_size
    )
    print("[DEBUG] Train DataLoader created")

    print("[DEBUG] Loading validation dataset...")
    t1 = time.time()
    val_dataset = CompressDataset(
        data_path=args.val_data_path,
        cube_size=args.val_cube_size,
        batch_size=args.batch_size
    )
    print(f"[DEBUG] Validation dataset initialized in {time.time() - t1:.2f}s")

    print("[DEBUG] Creating validation DataLoader...")
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size
    )
    print("[DEBUG] Validation DataLoader created")

    # ------------------------
    # CHECKPOINT SETUP
    # ------------------------
    str_time = datetime.now().isoformat()
    print('[DEBUG] Experiment Time:', str_time)
    checkpoint_dir = os.path.join(args.output_path, str_time, 'ckpt')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print('[DEBUG] Checkpoint directory created at', checkpoint_dir)

    # ------------------------
    # MODEL SETUP
    # ------------------------
    print("[DEBUG] Creating AutoEncoder model...")
    model = AutoEncoder(args)
    model = model.cuda()
    print('[DEBUG] Model moved to GPU')

    # ------------------------
    # OPTIMIZERS
    # ------------------------
    print("[DEBUG] Setting up optimizers and scheduler...")
    parameters = set(p for n, p in model.named_parameters() if not n.endswith(".quantiles"))
    optimizer = optim.Adam(parameters, lr=args.lr)

    scheduler_steplr = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.gamma)

    aux_parameters = set(p for n, p in model.named_parameters() if n.endswith(".quantiles"))
    aux_optimizer = optim.Adam(aux_parameters, lr=args.aux_lr)
    print("[DEBUG] Optimizers and scheduler ready")

    # ------------------------
    # RESUME TRAINING (if checkpoint provided)
    # ------------------------
    start_epoch = 0
    best_val_chamfer_loss = float('inf')

    if args.resume_path is not None and os.path.exists(args.resume_path):
        print(f"[DEBUG] Resuming training from checkpoint: {args.resume_path}")
        checkpoint = torch.load(args.resume_path, map_location='cuda')

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler_steplr.load_state_dict(checkpoint['scheduler_state_dict'])
            best_val_chamfer_loss = checkpoint.get('best_val_chamfer_loss', float('inf'))
            start_epoch = checkpoint['epoch']
            print(f"[DEBUG] Resumed from epoch {start_epoch} with best val loss {best_val_chamfer_loss:.6f}")
        else:
            model.load_state_dict(checkpoint)
            print("[DEBUG] Loaded model weights only (no optimizer state).")

    else:
        print("[DEBUG] No checkpoint provided, starting from scratch.")

    # ------------------------
    # TRAINING LOOP
    # ------------------------
    print("[DEBUG] Entering training loop...")

    for epoch in range(start_epoch, args.epochs):
        print(f"Starting epoch {epoch+1} / {args.epochs}")

        epoch_loss = AverageMeter()
        epoch_chamfer_loss = AverageMeter()
        epoch_density_loss = AverageMeter()
        epoch_pts_num_loss = AverageMeter()
        epoch_latent_xyzs_loss = AverageMeter()
        epoch_normal_loss = AverageMeter()
        epoch_bpp_loss = AverageMeter()
        epoch_aux_loss = AverageMeter()

        model.train()

        for i, input_dict in enumerate(train_loader):
            # input: (b, n, c)
            input = input_dict['xyzs'].cuda()
            # input: (b, c, n)
            input = input.permute(0, 2, 1).contiguous()

            # compress normal
            if args.compress_normal:
                normals = input_dict['normals'].cuda().permute(0, 2, 1).contiguous()
                input = torch.cat((input, normals), dim=1)

            # model forward
            decompressed_xyzs, loss, loss_items, bpp = model(input)
            epoch_loss.update(loss.item())
            epoch_chamfer_loss.update(loss_items['chamfer_loss'])
            epoch_density_loss.update(loss_items['density_loss'])
            epoch_pts_num_loss.update(loss_items['pts_num_loss'])
            epoch_latent_xyzs_loss.update(loss_items['latent_xyzs_loss'])
            epoch_normal_loss.update(loss_items['normal_loss'])
            epoch_bpp_loss.update(loss_items['bpp_loss'])

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update entropy bottleneck
            aux_loss = model.feats_eblock.loss()
            if args.quantize_latent_xyzs:
                aux_loss += model.xyzs_eblock.loss()
            epoch_aux_loss.update(aux_loss.item())

            aux_optimizer.zero_grad()
            aux_loss.backward()
            aux_optimizer.step()

            # print loss
            if (i + 1) % args.print_freq == 0:
                print("train epoch: %d/%d, iters: %d/%d, loss: %f, avg chamfer loss: %f, "
                      "avg density loss: %f, avg pts num loss: %f, avg latent xyzs loss: %f, "
                      "avg normal loss: %f, avg bpp loss: %f, avg aux loss: %f" %
                      (epoch + 1, args.epochs, i + 1, len(train_loader), epoch_loss.get_avg(),
                       epoch_chamfer_loss.get_avg(), epoch_density_loss.get_avg(),
                       epoch_pts_num_loss.get_avg(), epoch_latent_xyzs_loss.get_avg(),
                       epoch_normal_loss.get_avg(), epoch_bpp_loss.get_avg(), epoch_aux_loss.get_avg()))

        scheduler_steplr.step()

        interval = time.time() - start
        print("train epoch: %d/%d, time: %d mins %.1f secs, loss: %f, avg chamfer loss: %f, "
              "avg density loss: %f, avg pts num loss: %f, avg latent xyzs loss: %f, "
              "avg normal loss: %f, avg bpp loss: %f, avg aux loss: %f" %
              (epoch + 1, args.epochs, interval / 60, interval % 60, epoch_loss.get_avg(),
               epoch_chamfer_loss.get_avg(), epoch_density_loss.get_avg(),
               epoch_pts_num_loss.get_avg(), epoch_latent_xyzs_loss.get_avg(),
               epoch_normal_loss.get_avg(), epoch_bpp_loss.get_avg(), epoch_aux_loss.get_avg()))

        # ------------------------
        # VALIDATION
        # ------------------------
        model.eval()
        val_chamfer_loss = AverageMeter()
        val_normal_loss = AverageMeter()
        val_bpp = AverageMeter()

        with torch.no_grad():
            for input_dict in val_loader:
                input = input_dict['xyzs'].cuda()
                input = input.permute(0, 2, 1).contiguous()

                if args.compress_normal:
                    normals = input_dict['normals'].cuda().permute(0, 2, 1).contiguous()
                    input = torch.cat((input, normals), dim=1)
                    args.in_fdim = 6

                gt_xyzs = input[:, :3, :].contiguous()

                decompressed_xyzs, loss, loss_items, bpp = model(input)
                d1, d2, _, _ = chamfer_dist(gt_xyzs.permute(0, 2, 1).contiguous(),
                                            decompressed_xyzs.permute(0, 2, 1).contiguous())
                chamfer_loss = d1.mean() + d2.mean()
                val_chamfer_loss.update(chamfer_loss.item())
                val_normal_loss.update(loss_items['normal_loss'])
                val_bpp.update(bpp.item())

        print("val epoch: %d/%d, val bpp: %f, val chamfer loss: %f, val normal loss: %f" %
              (epoch + 1, args.epochs, val_bpp.get_avg(),
               val_chamfer_loss.get_avg(), val_normal_loss.get_avg()))

        # ------------------------
        # SAVE CHECKPOINT
        # ------------------------
        cur_val_chamfer_loss = val_chamfer_loss.get_avg()
        if cur_val_chamfer_loss < best_val_chamfer_loss or (epoch + 1) % args.save_freq == 0:
            model_name = 'ckpt-best.pth' if cur_val_chamfer_loss < best_val_chamfer_loss else f'ckpt-epoch-{epoch + 1:02d}.pth'
            model_path = os.path.join(checkpoint_dir, model_name)

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler_steplr.state_dict(),
                'best_val_chamfer_loss': best_val_chamfer_loss
            }, model_path)

            if cur_val_chamfer_loss < best_val_chamfer_loss:
                best_val_chamfer_loss = cur_val_chamfer_loss
                print(f"[DEBUG] Best model updated with val loss {best_val_chamfer_loss:.6f}")


def reset_model_args(train_args, model_args):
    """Copy training arguments to model arguments"""
    for arg in vars(train_args):
        setattr(model_args, arg, getattr(train_args, arg))


def parse_train_args():
    """Parse all arguments including dataset-specific and training-specific ones"""
    parser = argparse.ArgumentParser(description='Training Arguments')
    
    # First, add the dataset selector
    parser.add_argument('--dataset', default='shapenet', type=str, help='shapenet or semantickitti')
    parser.add_argument('--resume_path', default=None, type=str, help='path to checkpoint to resume from')
    
    # Add all possible arguments from both dataset parsers
    # Optimizer
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--aux_lr', default=1e-3, type=float, help='learning rate for entropy model')
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), type=float)
    
    # LR scheduler
    parser.add_argument('--lr_decay_step', default=15, type=int)
    parser.add_argument('--gamma', default=0.5, type=float)
    
    # Dataset paths
    parser.add_argument('--train_data_path', default=None, type=str)
    parser.add_argument('--train_cube_size', default=None, type=int)
    parser.add_argument('--val_data_path', default=None, type=str)
    parser.add_argument('--val_cube_size', default=None, type=int)
    parser.add_argument('--test_data_path', default=None, type=str)
    parser.add_argument('--test_cube_size', default=None, type=int)
    parser.add_argument('--peak', default=None, type=float)
    
    # Training
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--print_freq', default=1000, type=int)
    parser.add_argument('--save_freq', default=10, type=int)
    parser.add_argument('--output_path', default='./output', type=str)
    
    # Compression
    parser.add_argument('--compress_normal', default=False, type=str2bool)
    parser.add_argument('--in_fdim', default=3, type=int)
    
    # Model architecture
    parser.add_argument('--k', default=16, type=int)
    parser.add_argument('--downsample_rate', default=[1/3, 1/3, 1/3], nargs='+', type=float)
    parser.add_argument('--max_upsample_num', default=[8, 8, 8], nargs='+', type=int)
    parser.add_argument('--layer_num', default=3, type=int)
    parser.add_argument('--dim', default=8, type=int)
    parser.add_argument('--hidden_dim', default=64, type=int)
    parser.add_argument('--ngroups', default=1, type=int)
    parser.add_argument('--quantize_latent_xyzs', default=True, type=str2bool)
    parser.add_argument('--latent_xyzs_conv_mode', default='mlp', type=str)
    parser.add_argument('--sub_point_conv_mode', default='mlp', type=str)
    
    # Loss coefficients
    parser.add_argument('--chamfer_coe', default=0.1, type=float)
    parser.add_argument('--pts_num_coe', default=5e-7, type=float)
    parser.add_argument('--normal_coe', default=1e-2, type=float)
    parser.add_argument('--bpp_lambda', default=5e-4, type=float)
    parser.add_argument('--mean_distance_coe', default=5e1, type=float)
    parser.add_argument('--density_coe', default=1e-4, type=float)
    parser.add_argument('--latent_xyzs_coe', default=1e-2, type=float)
    
    # Test/evaluation
    parser.add_argument('--model_path', default='path to ckpt', type=str)
    parser.add_argument('--density_radius', default=0.15, type=float)
    parser.add_argument('--dist_coe', default=1e-5, type=float)
    parser.add_argument('--omega_xyzs', default=0.5, type=float)
    parser.add_argument('--omega_normals', default=0.5, type=float)
    
    args = parser.parse_args()
    
    # Set dataset-specific defaults
    if args.dataset == 'semantickitti':
        if args.train_data_path is None:
            args.train_data_path = './data/semantickitti/semantickitti_train_cube_size_12.pkl'
        if args.train_cube_size is None:
            args.train_cube_size = 12
        if args.val_data_path is None:
            args.val_data_path = './data/semantickitti/semantickitti_val_cube_size_12.pkl'
        if args.val_cube_size is None:
            args.val_cube_size = 12
        if args.test_data_path is None:
            args.test_data_path = './data/semantickitti/semantickitti_test_cube_size_12.pkl'
        if args.test_cube_size is None:
            args.test_cube_size = 12
    elif args.dataset == 'shapenet':
        if args.train_data_path is None:
            args.train_data_path = './data/shapenet/shapenet_train.pkl'
        if args.train_cube_size is None:
            args.train_cube_size = 1
        if args.val_data_path is None:
            args.val_data_path = './data/shapenet/shapenet_val.pkl'
        if args.val_cube_size is None:
            args.val_cube_size = 1
        if args.test_data_path is None:
            args.test_data_path = './data/shapenet/shapenet_test.pkl'
        if args.test_cube_size is None:
            args.test_cube_size = 1
    
    return args


if __name__ == "__main__":
    args = parse_train_args()
    assert args.dataset in ['shapenet', 'semantickitti']
    train(args)