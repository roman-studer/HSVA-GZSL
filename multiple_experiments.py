import uuid

### execute this function to train and test the vae-model

from HSVA_original import Model
import numpy as np
import pickle
import torch
import os
import argparse
import time

from wandb_logging import WandBLogger

import warnings
warnings.filterwarnings("ignore")

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='POLLEN', help='dataset to use')
parser.add_argument('--num_shots', type=int, default=0, help='number of shots')
parser.add_argument('--generalized', type=str2bool, default=True, help='generalized or not')
parser.add_argument('--cls_train_steps', type=int, default=200)
parser.add_argument('--subset', type=bool, default=True, help='use subset of data, only available for POLLEN')
parser.add_argument('--device', type=str, default='cuda', help='device to use')
parser.add_argument('--run_name', type=str, default=str(uuid.uuid4().hex[:8]), help='run name')
parser.add_argument('--random_state', type=int, default=42, help='random state')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=200, help='batch size')
parser.add_argument('--latent_size', type=int, default=15, help='latent size')
parser.add_argument('--lr_gen_model', type=float, default=0.000015, help='learning rate for generator model')
parser.add_argument('--lr_cls', type=float, default=0.00001, help='learning rate for classifier')
parser.add_argument('--beta_factor', type=float, default=0.25, help='beta factor')
parser.add_argument('--cross_reconstruction_factor', type=float, default=2.37, help='cross reconstruction factor')
parser.add_argument('--distance_factor', type=float, default=8.13, help='distance factor')
parser.add_argument('--loss', type=str, default='l2', help='loss to use, l1 or l2')
parser.add_argument('--beta_factor_end_epoch', type=int, default=93, help='beta factor end epoch')
parser.add_argument('--cross_reconstruction_factor_end_epoch', type=int, default=75, help='cross reconstruction factor end epoch')
parser.add_argument('--distance_factor_end_epoch', type=int, default=22, help='distance factor end epoch')
parser.add_argument('--coarse_latent_size', type=int, default=15, help='coarse latent size')
parser.add_argument('--recon_x_cyc_w', type=float, default=0.5, help='reconstruction weight')
parser.add_argument('--adapt_mode', type=str, default='SWD', help='adapt mode')
args = parser.parse_args()

if args.device == 'cuda':
    torch.cuda.manual_seed_all(args.random_state)
else:
    torch.manual_seed(args.random_state)


########################################
# the basic hyperparameters
########################################
hyperparameters = {
    'num_shots': args.num_shots,
    'device': args.device,
    'model_specifics': {'cross_reconstruction': True,
                       'name': 'CADA',
                       'distance': 'wasserstein',
                       'warmup': {'beta': {'factor': args.beta_factor,
                                           'end_epoch': args.beta_factor_end_epoch,
                                           'start_epoch': 0},
                                  'cross_reconstruction': {'factor': args.cross_reconstruction_factor,
                                                           'end_epoch': args.cross_reconstruction_factor_end_epoch,
                                                           'start_epoch': 21},
                                  'distance': {'factor': args.distance_factor,
                                               'end_epoch': args.distance_factor_end_epoch,
                                               'start_epoch': 0}}},

    'lr_gen_model': args.lr_gen_model,
    'generalized': args.generalized,
    'batch_size': args.batch_size,
    'samples_per_class': {'SUN': (200, 0, 400, 0),
                          'APY': (200, 0, 400, 0),
                          'CUB': (200, 0, 400, 0),
                          'AWA2': (200, 0, 400, 0),
                          'FLO': (200, 0, 400, 0),
                          'AWA1': (200, 0, 400, 0),
                          'POLLEN': (200, 0, 400, 0)},
    'epochs': args.n_epochs,
    'loss': args.loss,
    'auxiliary_data_source' : 'attributes',
    'lr_cls': args.lr_cls,
    'dataset': args.dataset,
    'hidden_size_rule': {'resnet_features': (42, 25),
                        'attributes': (50, 25),
                        'sentences': (50, 25) },
    'coarse_latent_size': args.coarse_latent_size,
    'latent_size': args.latent_size,  # 64 for CUB,AWA; 128 for SUN
    'recon_x_cyc_w': args.recon_x_cyc_w,
    'adapt_mode': args.adapt_mode,  # MCD or SWD
    'classifier': 'softmax',  # softmax
    'result_root': os.getcwd() + '/model/result',
    'subset': args.subset,
}

# The training epochs for the final classifier, for early stopping,
# as determined on the validation spit

cls_train_steps = [
      {'dataset': 'SUN',  'num_shots': 0, 'generalized': True, 'cls_train_steps': 21},
      {'dataset': 'SUN',  'num_shots': 0, 'generalized': False, 'cls_train_steps': 30},
      {'dataset': 'SUN',  'num_shots': 1, 'generalized': True, 'cls_train_steps': 22},
      {'dataset': 'SUN',  'num_shots': 1, 'generalized': False, 'cls_train_steps': 96},
      {'dataset': 'SUN',  'num_shots': 5, 'generalized': True, 'cls_train_steps': 29},
      {'dataset': 'SUN',  'num_shots': 5, 'generalized': False, 'cls_train_steps': 78},
      {'dataset': 'SUN',  'num_shots': 2, 'generalized': True, 'cls_train_steps': 29},
      {'dataset': 'SUN',  'num_shots': 2, 'generalized': False, 'cls_train_steps': 61},
      {'dataset': 'SUN',  'num_shots': 10, 'generalized': True, 'cls_train_steps': 79},
      {'dataset': 'SUN',  'num_shots': 10, 'generalized': False, 'cls_train_steps': 94},
      {'dataset': 'AWA1', 'num_shots': 0, 'generalized': True, 'cls_train_steps': 33},
      {'dataset': 'AWA1', 'num_shots': 0, 'generalized': False, 'cls_train_steps': 25},
      {'dataset': 'AWA1', 'num_shots': 1, 'generalized': True, 'cls_train_steps': 40},
      {'dataset': 'AWA1', 'num_shots': 1, 'generalized': False, 'cls_train_steps': 81},
      {'dataset': 'AWA1', 'num_shots': 5, 'generalized': True, 'cls_train_steps': 89},
      {'dataset': 'AWA1', 'num_shots': 5, 'generalized': False, 'cls_train_steps': 62},
      {'dataset': 'AWA1', 'num_shots': 2, 'generalized': True, 'cls_train_steps': 56},
      {'dataset': 'AWA1', 'num_shots': 2, 'generalized': False, 'cls_train_steps': 59},
      {'dataset': 'AWA1', 'num_shots': 10, 'generalized': True, 'cls_train_steps': 100},
      {'dataset': 'AWA1', 'num_shots': 10, 'generalized': False, 'cls_train_steps': 50},
      {'dataset': 'CUB',  'num_shots': 0, 'generalized': True, 'cls_train_steps': 23},
      {'dataset': 'CUB',  'num_shots': 0, 'generalized': False, 'cls_train_steps': 22},
      {'dataset': 'CUB',  'num_shots': 1, 'generalized': True, 'cls_train_steps': 34},
      {'dataset': 'CUB',  'num_shots': 1, 'generalized': False, 'cls_train_steps': 46},
      {'dataset': 'CUB',  'num_shots': 5, 'generalized': True, 'cls_train_steps': 64},
      {'dataset': 'CUB',  'num_shots': 5, 'generalized': False, 'cls_train_steps': 73},
      {'dataset': 'CUB',  'num_shots': 2, 'generalized': True, 'cls_train_steps': 39},
      {'dataset': 'CUB',  'num_shots': 2, 'generalized': False, 'cls_train_steps': 31},
      {'dataset': 'CUB',  'num_shots': 10, 'generalized': True, 'cls_train_steps': 85},
      {'dataset': 'CUB',  'num_shots': 10, 'generalized': False, 'cls_train_steps': 67},
      {'dataset': 'AWA2', 'num_shots': 0, 'generalized': True, 'cls_train_steps': 50},
      {'dataset': 'AWA2', 'num_shots': 0, 'generalized': False, 'cls_train_steps': 39},
      {'dataset': 'AWA2', 'num_shots': 1, 'generalized': True, 'cls_train_steps': 44},
      {'dataset': 'AWA2', 'num_shots': 1, 'generalized': False, 'cls_train_steps': 96},
      {'dataset': 'AWA2', 'num_shots': 5, 'generalized': True, 'cls_train_steps': 99},
      {'dataset': 'AWA2', 'num_shots': 5, 'generalized': False, 'cls_train_steps': 29},
      {'dataset': 'AWA2', 'num_shots': 2, 'generalized': True, 'cls_train_steps': 69},
      {'dataset': 'AWA2', 'num_shots': 2, 'generalized': False, 'cls_train_steps': 79},
      {'dataset': 'AWA2', 'num_shots': 10, 'generalized': True, 'cls_train_steps': 86},
      {'dataset': 'AWA2', 'num_shots': 10, 'generalized': False, 'cls_train_steps': 78},
      {'dataset': 'APY',  'num_shots': 0, 'generalized': True, 'cls_train_steps': 23},
      {'dataset': 'APY',  'num_shots': 0, 'generalized': False, 'cls_train_steps': 22},
      {'dataset': 'APY',  'num_shots': 1, 'generalized': True, 'cls_train_steps': 34},
      {'dataset': 'APY',  'num_shots': 1, 'generalized': False, 'cls_train_steps': 46},
      {'dataset': 'APY',  'num_shots': 5, 'generalized': True, 'cls_train_steps': 64},
      {'dataset': 'APY',  'num_shots': 5, 'generalized': False, 'cls_train_steps': 73},
      {'dataset': 'APY',  'num_shots': 2, 'generalized': True, 'cls_train_steps': 39},
      {'dataset': 'APY',  'num_shots': 2, 'generalized': False, 'cls_train_steps': 31},
      {'dataset': 'APY',  'num_shots': 10, 'generalized': True, 'cls_train_steps': 85},
      {'dataset': 'APY',  'num_shots': 10, 'generalized': False, 'cls_train_steps': 67},
      {'dataset': 'FLO',  'num_shots': 0, 'generalized': True, 'cls_train_steps': 23},
      {'dataset': 'FLO',  'num_shots': 0, 'generalized': False, 'cls_train_steps': 22},
      {'dataset': 'FLO',  'num_shots': 1, 'generalized': True, 'cls_train_steps': 34},
      {'dataset': 'FLO',  'num_shots': 1, 'generalized': False, 'cls_train_steps': 46},
      {'dataset': 'FLO',  'num_shots': 5, 'generalized': True, 'cls_train_steps': 64},
      {'dataset': 'FLO',  'num_shots': 5, 'generalized': False, 'cls_train_steps': 73},
      {'dataset': 'FLO',  'num_shots': 2, 'generalized': True, 'cls_train_steps': 39},
      {'dataset': 'FLO',  'num_shots': 2, 'generalized': False, 'cls_train_steps': 31},
      {'dataset': 'FLO',  'num_shots': 10, 'generalized': True, 'cls_train_steps': 85},
      {'dataset': 'FLO',  'num_shots': 10, 'generalized': False, 'cls_train_steps': 67},
      {'dataset': 'POLLEN', 'num_shots': 0, 'generalized': True, 'cls_train_steps': args.cls_train_steps},
      {'dataset': 'POLLEN', 'num_shots': 0, 'generalized': True, 'cls_train_steps': args.cls_train_steps},
      {'dataset': 'POLLEN', 'num_shots': 4, 'generalized': True, 'cls_train_steps': args.cls_train_steps},
]

##################################
# change some hyperparameters here
##################################
hyperparameters['dataset'] = args.dataset
hyperparameters['num_shots'] = args.num_shots
hyperparameters['generalized'] = args.generalized

hyperparameters['cls_train_steps'] = [x['cls_train_steps']  for x in cls_train_steps
                                        if all([hyperparameters['dataset']==x['dataset'],
                                        hyperparameters['num_shots']==x['num_shots'],
                                        hyperparameters['generalized']==x['generalized'] ])][0]

print('***')
print(hyperparameters['cls_train_steps'] )
if hyperparameters['generalized']:
    if hyperparameters['num_shots']==0:
        hyperparameters['samples_per_class'] = {'CUB': (200, 0, 400, 0), 'SUN': (200, 0, 400, 0),
                                'APY': (200, 0,  400, 0), 'AWA1': (200, 0, 400, 0),
                                'AWA2': (200, 0, 400, 0), 'FLO': (200, 0, 400, 0),
                                'POLLEN': (200, 0, 400, 0)}
    else:
        hyperparameters['samples_per_class'] = {'CUB': (200, 0, 200, 200), 'SUN': (200, 0, 200, 200),
                                                    'APY': (200, 0, 200, 200), 'AWA1': (200, 0, 200, 200),
                                                    'AWA2': (200, 0, 200, 200), 'FLO': (200, 0, 200, 200),
                                                    'POLLEN': (200, 0, 400, 0)}
else:
    if hyperparameters['num_shots']==0:
        hyperparameters['samples_per_class'] = {'CUB': (0, 0, 400, 0), 'SUN': (0, 0, 200, 0),
                                                    'APY': (0, 0, 200, 0), 'AWA1': (0, 0, 800, 0),
                                                    'AWA2': (0, 0, 200, 0), 'FLO': (0, 0, 200, 0),
                                                    'POLLEN': (200, 0, 400, 0)}
    else:
        hyperparameters['samples_per_class'] = {'CUB': (0, 0, 200, 200), 'SUN': (0, 0, 200, 200),
                                                    'APY': (0, 0, 200, 200), 'AWA1': (0, 0, 200, 200),
                                                    'AWA2': (0, 0, 200, 200), 'FLO': (0, 0, 200, 200),
                                                    'POLLEN': (200, 0, 400, 0)}



for i in range(10):
    hyperparameters['split'] = 1

    model = Model(hyperparameters)
    model.to(hyperparameters['device'])

    """
    ########################################
    ### load model where u left
    ########################################
    saved_state = torch.load('./saved_models/CADA_trained.pth.tar')
    model.load_state_dict(saved_state['state_dict'])
    for d in model.all_data_sources_without_duplicates:
        model.encoder[d].load_state_dict(saved_state['encoder'][d])
        model.decoder[d].load_state_dict(saved_state['decoder'][d])
    ########################################
    """
    logger = WandBLogger(run_name=str(i) + '_' + args.run_name)

    logger.log_config(hyperparameters)

    model.train_vae(logger=logger)

    logger.finish()

    del model
    del logger
