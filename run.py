import torch
from flerken.framework.framework import Experiment

import VnBSS
import config
from default import *

args = argparse_default()

cfg_path = config.__path__[0]

# Defining the experiment
ex = Experiment(args.arxiv_path, args.workname)
if ex.resume_from is None:
    files, names = ex.IO.load_cfg(cfg_path)
    for file, name in zip(files, names):
        ex.IO.add_cfg(name, file)
    ex.IO.add_cfg('argparse', args.__dict__)
# Appending the whole package code to the tracker
ex.IO.add_package(VnBSS)
ex.IO.add_package(config)
# Loading the model
device = torch.device(args.device)
config.test_cfg['remix_input'] = args.remix
test_cfg = config.test_cfg if args.testing else {}
net_cfg = ex.legacy_cfg if args.legacy else ex.net_cfg
iter_param, model, model_kwargs = VnBSS.ModelConstructor(
    debug=DEBUG,
    **config.fourier_defaults,
    **ex.constructor_defaults,
).prepare(args.model).update(**net_cfg, **test_cfg,
                             device=device,
                             multitask=ex.hyptrs.multitask,
                             white_metrics=args.white_metrics).build()

DUMP_FILES['force'] = args.dumping
for loudness in args.loudness_levels:
    DUMP_FILES[f'val_seen_{loudness}'] = {'enabled': True, 'iter_freq': 50, 'epoch_freq': 4}
    DUMP_FILES[f'test_seen_{loudness}'] = {'enabled': True, 'iter_freq': 50, 'epoch_freq': 4}

for test_subset in args.test_in:
    for loudness in args.loudness_levels:
        DUMP_FILES[f'{test_subset}_{loudness}'] = {'enabled': True, 'iter_freq': 50, 'epoch_freq': 4}

trainer = VnBSS.Trainer(device, model, dataparallel=False, input_shape=None,
                        debug=DEBUG,
                        multitask=ex.hyptrs.multitask,
                        n_epochs=ex.hyptrs.epochs,
                        criterion=VnBSS.MultiTaskLoss(),
                        initializable_layers=None,
                        dump_iteration_files=DUMP_FILES,
                        white_metrics=args.white_metrics)

if args.pretrained_from is not None:
    trainer._model.load(args.pretrained_from, strict_loading=True, from_checkpoint=True)

trainer.model.to(device)
trainer.optimizer = torch.optim.SGD(iter_param, **ex.optimizer)

with ex.autoconfig(trainer) as trainer:
    # Dataloader pattern instance from which arises all the dataloaders
    n_iter = config.VAL_ITERATIONS if args.testing else config.TRAIN_ITERATIONS
    dl_pattern = VnBSS.DataloaderConstructor(obj=None,
                                             debug=DEBUG,
                                             n_iterations=n_iter,
                                             batch_size=ex.hyptrs['batch_size'],
                                             dataset_paths=ex.dataset_paths,
                                             **config.dataloader_constructor_defaults)
    trainer.loss_['train']['iter'].enabled = True
    trainer.loss_['val']['iter'].enabled = True
    if args.testing:

        trainer.epoch = 0
        with torch.no_grad():
            for test_subset in args.test_in:
                test_dl = VnBSS.DataloaderConstructor(dl_pattern, trace_init=ex.trace_paths[test_subset])
                test_dl.set_mode(test_subset).add_acappella(mouth_shape=ex.hyptrs.mouth_shape,
                                                            crop_mouth=ex.hyptrs.crop_mouth,
                                                            savgol_kwargs=ex.hyptrs.savgol_kwargs,
                                                            multitask=ex.hyptrs.multitask,
                                                            is_enabled=trainer.model.enabled,
                                                            **ex.dataset_exclusion)
                test_dl, _ = test_dl.add_audioset(n_sources=ex.hyptrs.n_sources).build(ex.hyptrs.balanced_sampling,
                                                                                       mean=config.MEAN,
                                                                                       std=config.STD)

                for loudness in args.loudness_levels:
                    test_dl.dataset.loudness_coef = loudness

                    trainer.run_epoch(test_dl, f'{test_subset}_{loudness}',
                                      backprop=False,
                                      metrics=['bss_eval'] if args.white_metrics else ['loss', 'bss_eval'],
                                      send=SEND,
                                      debug=DEBUG)
    else:
        val_seen_dl = VnBSS.DataloaderConstructor(dl_pattern, n_iterations=config.VAL_ITERATIONS).set_mode('val_seen')
        val_seen_dl.add_acappella(mouth_shape=ex.hyptrs.mouth_shape,
                                  crop_mouth=ex.hyptrs.crop_mouth,
                                  savgol_kwargs=ex.hyptrs.savgol_kwargs,
                                  multitask=ex.hyptrs.multitask,
                                  is_enabled=trainer.model.enabled,
                                  **ex.dataset_exclusion)
        val_seen_dl, _ = val_seen_dl.add_audioset(n_sources=ex.hyptrs.n_sources).build(ex.hyptrs.balanced_sampling,
                                                                                       mean=config.MEAN,
                                                                                       std=config.STD)

        train_dl = VnBSS.DataloaderConstructor(dl_pattern, n_iterations=config.TRAIN_ITERATIONS).set_mode('train')

        train_dl.set_mode('train').add_acappella(mouth_shape=ex.hyptrs.mouth_shape,
                                                 crop_mouth=ex.hyptrs.crop_mouth,
                                                 savgol_kwargs=ex.hyptrs.savgol_kwargs,
                                                 multitask=ex.hyptrs.multitask,
                                                 is_enabled=trainer.model.enabled,
                                                 **ex.dataset_exclusion)
        train_dl, _ = train_dl.add_audioset(n_sources=ex.hyptrs.n_sources).build(ex.hyptrs.balanced_sampling,
                                                                                 mean=config.MEAN,
                                                                                 std=config.STD)
        for trainer.epoch in range(trainer.start_epoch, trainer.EPOCHS):
            train_dl.dataset.loudness_coef = args.loudness_train
            trainer.run_epoch(train_dl, f'train',
                              backprop=True,
                              metrics=['loss'],
                              send=SEND,
                              debug=DEBUG)
            if not ex.hyptrs.overfit:
                with torch.no_grad():
                    for loudness in args.loudness_levels:
                        val_seen_dl.dataset.loudness_coef = loudness
                        trainer.run_epoch(val_seen_dl, f'val_seen_{loudness}',
                                          backprop=False,
                                          metrics=['loss', 'bss_eval'],
                                          checkpoint=trainer.checkpoint(metric='loss', freq=1),
                                          send=SEND,
                                          debug=DEBUG)
