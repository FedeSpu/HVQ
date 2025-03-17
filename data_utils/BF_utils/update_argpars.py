import os
import os.path as ops

from hvq.utils.arg_pars import opt
from hvq.utils.util_functions import update_opt_str, dir_check
from hvq.utils.logging_setup import path_logger
import torch


def update():

    opt.data = ops.join(opt.dataset_root, 'features')
    opt.gt = ops.join(opt.dataset_root, 'groundTruth')
    opt.output_dir = ops.join(opt.dataset_root, 'output')
    opt.mapping_dir = ops.join(opt.dataset_root, 'mapping')
    dir_check(opt.output_dir)
    if torch.cuda.is_available():
        opt.device = 'cuda'

    opt.bg = False  # YTI argument
    if opt.model_name == 'nothing':
        opt.load_embed_feat = True

    update_opt_str()

    logger = path_logger()

    vars_iter = list(vars(opt))
    for arg in sorted(vars_iter):
        logger.debug('%s: %s' % (arg, getattr(opt, arg)))

