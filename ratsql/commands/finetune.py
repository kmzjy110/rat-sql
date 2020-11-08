import argparse
import collections
import datetime
import json
import os
import sys

import _jsonnet
import attr
import torch
import tqdm
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
# noinspection PyUnresolvedReferences
from ratsql import beam_search

from ratsql.utils import registry
from ratsql.utils import random_state
from ratsql.utils import saver as saver_mod

# noinspection PyUnresolvedReferences
from ratsql.utils import vocab
from ratsql.commands.train import Logger

from ratsql.models.spider import spider_beam_search
def add_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', required=True)
    parser.add_argument('--finetunedir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-args')
    parser.add_argument('--infer-output-path', required=True)

    parser.add_argument('--beam-size', required=True, type=int)
    parser.add_argument('--output-history', action='store_true')

    parser.add_argument('--use_heuristic', action='store_true')

    args = parser.parse_args()
    return args


@attr.s
class FineTuneConfig:
    eval_every_n = attr.ib(default=100)
    report_every_n = attr.ib(default=10)
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
        self.config=config
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
    def _eval_model(logger, model, last_step, eval_data, eval_section, report_every_n):
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
        # if last_step % report_every_n ==0:
        #     kv_stats = ", ".join(f"{k} = {v}" for k, v in stats.items())
        #     logger.log(f"Step {last_step} stats, {eval_section}: {kv_stats}")
        return stats

    def _infer_one(self, model, data_item, preproc_item, beam_size, output_history=False, use_heuristic=True):
        if use_heuristic:
            # TODO: from_cond should be true from non-bert model
            beams = spider_beam_search.beam_search_with_heuristics(
                model, data_item, preproc_item, beam_size=beam_size, max_steps=1000, from_cond=False)
        else:
            beams = beam_search.beam_search(
                model, data_item, preproc_item, beam_size=beam_size, max_steps=1000)
        decoded = []
        for beam in beams:
            model_output, inferred_code = beam.inference_state.finalize()

            decoded.append({
                'orig_question': data_item.orig["question"],
                'model_output': model_output,
                'inferred_code': inferred_code,
                'score': beam.score,
                **({
                       'choice_history': beam.choice_history,
                       'score_history': beam.score_history,
                   } if output_history else {})})
        return decoded

    @staticmethod
    def _yield_batches_from_epochs(loader):
        while True:
            for batch in loader:
                yield batch
    def finetune(self, config, model_load_dir, model_save_dir, infer_output_path, beam_size, output_history,
                 use_heuristic):
        #random_seeds = [i for i in range(10)]
        orig_data = registry.construct('dataset', self.config['data']['val'])
        databases = orig_data.get_databases()
        random_seeds = [0]

        for seed in random_seeds:
            data_random = random_state.RandomContext(seed)
            print("seed:", seed)
            metrics_list = []
            scores = []
            with data_random:
                for database in databases:
                    current_infer_output_path = infer_output_path+"/"+database
                    os.makedirs(os.path.dirname(current_infer_output_path), exist_ok=True)
                    infer_output = open(current_infer_output_path, 'w')

                    spider_data = registry.construct('dataset',  self.config['data']['val'], database =database)
                    val_data = self.model_preproc.dataset('val', database=database)
                    val_data_loader = self._yield_batches_from_epochs(torch.utils.data.DataLoader(val_data, batch_size=1, collate_fn=lambda x:x,
                                                                  shuffle=False))
                    assert len(val_data) == len(spider_data)
                    if len(val_data)==0:
                        continue
                    print("database:", database)
                    #TODO: RANDOMIZE DATA
                    optimizer, lr_scheduler = self.construct_optimizer_and_lr_scheduler(config)
                    saver = saver_mod.Saver(
                        {"model": self.model, "optimizer": optimizer}, keep_every_n=self.finetune_config.keep_every_n)
                    last_step = saver.restore(model_load_dir, map_location=self.device)
                    self.logger.log(f"Loaded trained model; last_step:{last_step}")
                    keyerror_flag = False
                    for i, (orig_item, preproc_item) in enumerate(
                            tqdm.tqdm(zip(spider_data, val_data),
                                      total=len(val_data))):
                        try:
                            decoded = self._infer_one(self.model, orig_item, preproc_item, beam_size, output_history,
                                                      use_heuristic)
                            with self.model_random:
                                loss = self.model.compute_loss(next(val_data_loader))
                                norm_loss = loss/self.finetune_config.num_batch_accumulated
                                norm_loss.backward()

                                if self.finetune_config.clip_grad:
                                    torch.nn.utils.clip_grad_norm_(optimizer.bert_param_group["params"], \
                                                                   self.finetune_config.clip_grad)
                                optimizer.step()
                                lr_scheduler.update_lr(last_step)
                                optimizer.zero_grad()
                            infer_output.write(
                                json.dumps({
                                    'index': i,
                                    'beams': decoded,
                                }) + '\n')
                            infer_output.flush()
                            # stats = self._eval_model(self.logger, self.model, last_step, batch, 'val',
                            #                          self.finetune_config.report_every_n)
                            # val_losses.append(stats['loss'])
                        except KeyError:
                            self.logger.log("keyError")
                            keyerror_flag=True
                            break
                    if not keyerror_flag:
                        inferred = open(current_infer_output_path)
                        metrics = spider_data.Metrics(spider_data)
                        inferred_lines = list(inferred)
                        if len(inferred_lines) < len(spider_data):
                            raise Exception(f'Not enough inferred: {len(inferred_lines)} vs {len(data)}')

                        for line in inferred_lines:
                            infer_results = json.loads(line)
                            if infer_results['beams']:
                                inferred_code = infer_results['beams'][0]['inferred_code']
                            else:
                                inferred_code = None
                            if 'index' in infer_results:
                                metrics.add(spider_data[infer_results['index']], inferred_code)
                            else:
                                metrics.add(None, inferred_code, obsolete_gold_code=infer_results['gold_code'])
                        final_metrics = metrics.finalize()
                        metrics_list.append(final_metrics)
                        print(final_metrics['total_scores']['all']['exact'])
                        scores.append((final_metrics['total_scores']['all']['exact'], len(spider_data)))
                #if last_step % self.finetune_config.save_every_n == 0:
                    #saver.save(model_save_dir+'/seed_'+seed, last_step)
            print('scores',scores)
            print("average score:", self.aggregate_score(scores))

    def aggregate_score(self, scores):
        total_num = 0
        total_score = 0
        for score, num_datapoint in scores:
            total_score += score * num_datapoint
            total_num += num_datapoint
        return total_score/total_num
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
        args.logdir = os.path.join(args.logdir, config['model_name'])

    # Initialize the logger
    reopen_to_flush = config.get('log', {}).get('reopen_to_flush')
    logger = Logger(os.path.join(args.finetunedir, 'finetunelog.txt'), reopen_to_flush)

    # Save the config info
    with open(os.path.join(args.finetunedir,
                           f'config-{datetime.datetime.now().strftime("%Y%m%dT%H%M%S%Z")}.json'), 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)

    logger.log(f'Logging to {args.finetunedir}')
    infer_output_path = args.infer_output_path
    print("infer_output_path:", args.infer_output_path)
    os.makedirs(os.path.dirname(infer_output_path), exist_ok=True)
    if os.path.exists(infer_output_path):
        print(f'Output file {infer_output_path} already exists')
        sys.exit(1)
    # Construct trainer and do training
    beam_size = args.beam_size
    output_history = args.output_history
    use_heuristic = args.use_heuristic
    finetuner = FineTuner(logger, config)
    finetuner.finetune(config, model_load_dir=args.logdir, model_save_dir=args.finetunedir,
                       infer_output_path=infer_output_path, beam_size = beam_size, output_history=output_history,
                       use_heuristic=use_heuristic)

if __name__ == '__main__':
    args = add_parser()
    main(args)