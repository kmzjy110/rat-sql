import argparse
import collections
import datetime
import json
import os

import _jsonnet
import attr
import torch

# These imports are needed for registry.lookup
# noinspection PyUnresolvedReferences
from ratsql import ast_util
# noinspection PyUnresolvedReferences
from ratsql import datasets
# noinspection PyUnresolvedReferences
from ratsql import grammars
# noinspection PyUnresolvedReferences
from ratsql import models
# noinspection PyUnresolvedReferences
from ratsql import optimizers

from ratsql.utils import registry
from ratsql.utils import random_state
from ratsql.utils import saver as saver_mod

# noinspection PyUnresolvedReferences
from ratsql.utils import vocab
from ratsql.commands.train import Logger
def add_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', required=True)
    parser.add_argument('--finetunedir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-args')
    args = parser.parse_args()
    return args


@attr.s
class FineTuneConfig:
    eval_every_n = attr.ib(default=100)
    report_every_n = attr.ib(default=100)
    save_every_n = attr.ib(default=100)
    keep_every_n = attr.ib(default=1000)

    batch_size = attr.ib(default=1)
    max_steps = attr.ib(default=100000)
    num_eval_items = attr.ib(default=None)
    eval_on_train = attr.ib(default=True)
    eval_on_val = attr.ib(default=True)

    # Seed for RNG used in shuffling the training data.
    data_seed = attr.ib(default=None)
    # Seed for RNG used in initializing the model.
    init_seed = attr.ib(default=None)
    # Seed for RNG used in computing the model's training loss.
    # Only relevant with internal randomness in the model, e.g. with dropout.
    model_seed = attr.ib(default=None)

    num_batch_accumulated = attr.ib(default=1)
    clip_grad = attr.ib(default=None)

class FineTuner:
    def __init__(self, logger, config):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.logger = logger
        self.finetune_config = registry.instantiate(FineTuneConfig, config['finetune'])
        self.model_random = random_state.RandomContext(self.finetune_config.model_seed)

        self.init_random = random_state.RandomContext(self.finetune_config.init_seed)
        with self.init_random:
            # 0. Construct preprocessors
            self.model_preproc = registry.instantiate(
                registry.lookup('model', config['model']).Preproc,
                config['model'],
                unused_keys=('name',))
            self.model_preproc.load()

            # 1. Construct model
            self.model = registry.construct('model', config['model'],
                                            unused_keys=('encoder_preproc', 'decoder_preproc'),
                                            preproc=self.model_preproc, device=self.device)
            self.model.to(self.device)
    @staticmethod
    def _eval_model(logger, model, last_step, eval_data, eval_section):
        stats = collections.defaultdict(float)
        model.eval()
        with torch.no_grad():
            batch_res = model.eval_on_batch(eval_data)
            for k, v in batch_res.items():
                stats[k] += v
        model.train()

        # Divide each stat by 'total'
        for k in stats:
            if k != 'total':
                stats[k] /= stats['total']
        if 'total' in stats:
            del stats['total']

        kv_stats = ", ".join(f"{k} = {v}" for k, v in stats.items())
        logger.log(f"Step {last_step} stats, {eval_section}: {kv_stats}")

    def finetune(self, config, model_load_dir, model_save_dir):



        random_seeds = [i for i in range(10)]
        for seed in random_seeds:
            data_random = random_state.RandomContext(seed)
            with data_random:

                val_data = self.model_preproc.dataset('val')
                val_data_loader = torch.utils.data.DataLoader(
                    val_data,
                    batch_size=self.finetune_config.eval_batch_size,
                    collate_fn=lambda x: x)
                optimizer, lr_scheduler = self.construct_optimizer_and_lr_scheduler(config)
                saver = saver_mod.Saver(
                    {"model": self.model, "optimizer": optimizer}, keep_every_n=self.finetune_config.keep_every_n)
                last_step = saver.restore(model_load_dir, map_location=self.device)
                self.logger.log("Loaded trained model; last_step:", last_step)

                for batch in val_data_loader:
                    self._eval_model(self.logger, self.model, last_step, batch, 'val')
                    with self.model_random:
                        loss = self.model.compute_loss(batch)
                        norm_loss = loss/self.finetune_config.num_batch_accumulated
                        norm_loss.backward()

                        if self.finetune_config.clip_grad:
                            torch.nn.utils.clip_grad_norm_(optimizer.bert_param_group["params"], \
                                                           self.finetune_config.clip_grad)
                        optimizer.step()
                        lr_scheduler.update_lr(last_step)
                        optimizer.zero_grad()
                    last_step+=1
                    self.logger.log("Stepped with val data. Step:", last_step)
                if last_step % self.finetune_config.save_every_n == 0:
                    saver.save(model_save_dir+'/seed_'+seed, last_step)
    def construct_optimizer_and_lr_scheduler(self, config):
        if config["optimizer"].get("name", None) == 'bertAdamw':
            bert_params = list(self.model.encoder.bert_model.parameters())
            assert len(bert_params) > 0
            non_bert_params = []
            for name, _param in self.model.named_parameters():
                if "bert" not in name:
                    non_bert_params.append(_param)
            assert len(non_bert_params) + len(bert_params) == len(list(self.model.parameters()))

            optimizer = registry.construct('optimizer', config['optimizer'], non_bert_params=non_bert_params,
                                           bert_params=bert_params)
            lr_scheduler = registry.construct('lr_scheduler',
                                              config.get('lr_scheduler', {'name': 'noop'}),
                                              param_groups=[optimizer.non_bert_param_group,
                                                            optimizer.bert_param_group])
        else:
            optimizer = registry.construct('optimizer', config['optimizer'], params=self.model.parameters())
            lr_scheduler = registry.construct('lr_scheduler',
                                              config.get('lr_scheduler', {'name': 'noop'}),
                                              param_groups=optimizer.param_groups)
        return optimizer,lr_scheduler

def main(args):
    if args.config_args:
        config = json.loads(_jsonnet.evaluate_file(args.config, tla_codes={'args': args.config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    if 'model_name' in config:
        args.finetunedir = os.path.join(args.finetunedir, config['model_name'])

    # Initialize the logger
    reopen_to_flush = config.get('log', {}).get('reopen_to_flush')
    logger = Logger(os.path.join(args.finetunedir, 'finetunelog.txt'), reopen_to_flush)

    # Save the config info
    with open(os.path.join(args.finetunedir,
                           f'config-{datetime.datetime.now().strftime("%Y%m%dT%H%M%S%Z")}.json'), 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)

    logger.log(f'Logging to {args.finetunedir}')

    # Construct trainer and do training
    finetuner = FineTuner(logger, config)
    finetuner.finetune(config, model_load_dir=args.logdir, model_save_dir=args.finetunedir)

if __name__ == '__main__':
    args = add_parser()
    main(args)