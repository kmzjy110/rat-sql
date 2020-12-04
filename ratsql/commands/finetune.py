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
import numpy as np
import matplotlib.pyplot as plt
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
from ratsql.models.enc_dec import ZippedDataset
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
        random_seeds = [i for i in range(3)]
        orig_data = registry.construct('dataset', self.config['data']['val'])
        databases = orig_data.get_databases()

        for seed in random_seeds:
            data_random = random_state.RandomContext(seed)
            print("seed:", seed)
            metrics_list = []
            batch_1_scores = []
            no_grad_scores = []
            batch_32_scores = []
            n_2_scores = []
            with data_random:
                # print("No grad")
                # no_grad_infer_output_path = infer_output_path + "no_grad/no_grad.infer"
                # os.makedirs(os.path.dirname(no_grad_infer_output_path), exist_ok=False)
                # print(no_grad_infer_output_path)
                # for database in databases:
                #
                #     self.finetune_on_database(no_grad_infer_output_path, database, config, model_load_dir,
                #                               beam_size, output_history, use_heuristic, metrics_list, no_grad_scores,
                #                               take_grad_steps=False, batch_size="1")
                # print("No grad scores", no_grad_scores)
                # print("average", self.aggregate_score(no_grad_scores))
                no_grad_scores = [('dog_kennels', 0.5, 82), ('flight_2', 0.5875, 80),
                                  ('pets_1', 0.4523809523809524, 42),
                                  ('concert_singer', 0.5333333333333333, 45),
                                  ('museum_visit', 0.4444444444444444, 18),
                                  ('battle_death', 0.5625, 16),
                                  ('student_transcripts_tracking', 0.48717948717948717, 78),
                                  ('singer', 0.7333333333333333, 30),
                                  ('cre_Doc_Template_Mgt', 0.7023809523809523, 84),
                                  ('world_1', 0.19166666666666668, 120),
                                  ('employee_hire_evaluation', 0.8421052631578947, 38),
                                  ('network_1', 0.6428571428571429, 56),
                                  ('poker_player', 0.875, 40), ('real_estate_properties', 0.25, 4),
                                  ('course_teach', 0.7333333333333333, 30),
                                  ('voter_1', 0.4666666666666667, 15), ('wta_1', 0.5, 62),
                                  ('orchestra', 0.85, 40), ('car_1', 0.32608695652173914, 92),
                                  ('tvshow', 0.6612903225806451, 62)]
                average = self.aggregate_score(no_grad_scores)
                no_grad_scores.append(("average", average))
                # self.plot(no_grad_scores, "no_grad_scores.png", "no grad scores")
                # print("No grad scores", no_grad_scores)
                # print("average", average)
                #
                # print("batch size 1")
                # batch_1_infer_output_path = infer_output_path + "seed_"+str(seed)+"/batch_1/batch_1.infer"
                # os.makedirs(os.path.dirname(batch_1_infer_output_path), exist_ok=False)
                # print(batch_1_infer_output_path)
                # for database in databases:
                #     self.finetune_on_database(batch_1_infer_output_path, database, config, model_load_dir,
                #                               beam_size, output_history, use_heuristic, metrics_list, batch_1_scores,
                #                               take_grad_steps=True, batch_size="1")
                # average = self.aggregate_score(batch_1_scores)
                # batch_1_scores.append(("average", average))
                # self.plot(batch_1_scores, "batch_1_scores_seed_"+str(seed)+".png", "batch size 1 scores seed "+ str(seed))
                # print("batch size 1 scores", batch_1_scores)
                # print("average", average)
                #
                # print("batch size 32")
                # batch_32_infer_output_path = infer_output_path + "seed_"+str(seed)+"/batch_32/batch_32.infer"
                # os.makedirs(os.path.dirname(batch_32_infer_output_path), exist_ok=False)
                # print(batch_32_infer_output_path)
                # for database in databases:
                #     self.finetune_on_database(batch_32_infer_output_path, database, config, model_load_dir,
                #                               beam_size, output_history, use_heuristic, metrics_list, batch_32_scores,
                #                               take_grad_steps=True, batch_size="32")
                # average = self.aggregate_score(batch_32_scores)
                # batch_32_scores.append(("average", average))
                # self.plot(batch_32_scores, "batch_32_scores_seed_"+str(seed)+".png", "batch size 32 scores seed " + str(seed))
                # print("batch size 32 scores", batch_32_scores)
                # print("average",average)

                print("n^2")
                n_2_infer_output_path = infer_output_path + "seed_"+str(seed)+"/n_2/n_2.infer"
                os.makedirs(os.path.dirname(n_2_infer_output_path), exist_ok=False)
                print(n_2_infer_output_path)
                for database in databases:
                    self.finetune_on_database(n_2_infer_output_path, database, config, model_load_dir,
                                              beam_size, output_history, use_heuristic, metrics_list, n_2_scores,
                                              take_grad_steps=True, batch_size="n^2")
                average = self.aggregate_score(n_2_scores)
                n_2_scores.append(("average", average))
                self.plot(n_2_scores, "n_2_scores_no_repeat_seed_"+str(seed)+".png", "batch n^2 scores no repeat seed " + str(seed))
                print("n^2 scores", n_2_scores)
                print("average", average)
                # print("Score on entire validation set:")
                # self.finetune_on_database(infer_output_path, None, config, model_load_dir,
                #                           beam_size, output_history, use_heuristic, metrics_list, scores, take_grad_steps=False)
                print("")
                print("changes")

                # print("batch size 1 changes")
                # self.plot(self.get_change(no_grad_scores, batch_1_scores),
                #           "batch_size_1_changes_seed_"+str(seed)+".png",
                #           "batch size 1 score changes")
                # print(self.get_change(no_grad_scores, batch_1_scores))
                #
                # print("batch size 32 changes")
                # self.plot(self.get_change(no_grad_scores, batch_32_scores),
                #           "batch_size_32_changes_seed_" + str(seed) + ".png",
                #           "batch size 32 score changes")
                # print(self.get_change(no_grad_scores, batch_32_scores))

                print("batch size n^2 changes")
                self.plot(self.get_change(no_grad_scores, n_2_scores),
                          "batch_size_n_2_no_repeat_changes_seed_" + str(seed) + ".png",
                          "batch size n^2 score changes with no repeat queries")
                print(self.get_change(no_grad_scores, n_2_scores))

    def plot(self, scores, filename, title='scores'):
        plt.figure()

        x = [item[0] for item in scores]
        y = [item[1] for item in scores]
        x_pos = [i for i, _ in enumerate(x)]
        plt.bar(x_pos, y)
        plt.xlabel('database')
        plt.ylabel(title)
        plt.xticks(x_pos, x, rotation=90)
        plt.tight_layout()
        plt.show()
        plt.savefig(filename)

    def get_change(self, no_grad_scores, new_scores):
        results = []
        for no_grad_score in no_grad_scores:
            for new_score in new_scores:
                if no_grad_score[0] == new_score[0]:
                    results.append((no_grad_score[0], new_score[1]-no_grad_score[1]))
        return results
    def get_no_repeat_data_indices(self, spider_data):
        seen = set()
        indices = []
        for i in range(len(spider_data)):
            if spider_data[i].orig.get('query').lower() not in seen:
                indices.append(i)
                seen.add(spider_data[i].orig.get('query').lower())
        return indices


    def finetune_on_database(self,infer_output_path, database, config,model_load_dir, beam_size, output_history,
                             use_heuristic, metrics_list, scores, take_grad_steps=True, batch_size="1"):
        if database:
            current_infer_output_path = infer_output_path + "/" + database
        else:
            current_infer_output_path = infer_output_path+"/"+"entire_val"
        os.makedirs(os.path.dirname(current_infer_output_path), exist_ok=True)
        infer_output = open(current_infer_output_path, 'w')

        spider_data = registry.construct('dataset', self.config['data']['val'], database=database)
        val_data = self.model_preproc.dataset('val', database=database)

        # val_data_loader = self._yield_batches_from_epochs(
        #     torch.utils.data.DataLoader(val_data, batch_size=1, collate_fn=lambda x: x,
        #                                 shuffle=False))

        assert len(val_data) == len(spider_data)
        if len(val_data) == 0:
            return
        if batch_size=="32":
            if len(val_data)<32:
                return
        print("database:", database)

        if batch_size=="n^2":
            indices = np.random.permutation(self.get_no_repeat_data_indices(spider_data))
            print("length of data:", len(val_data))
            print("length of data after removing repeat entries:", len(indices))
        else:
            indices = np.random.permutation(len(val_data))

        # TODO: RANDOMIZE DATA
        optimizer, lr_scheduler = self.construct_optimizer_and_lr_scheduler(config)
        saver = saver_mod.Saver(
            {"model": self.model, "optimizer": optimizer}, keep_every_n=self.finetune_config.keep_every_n)
        last_step = saver.restore(model_load_dir, map_location=self.device)
        self.logger.log(f"Loaded trained model; last_step:{last_step}")
        current_batch = []
        clear_batch=False
        current_number = 0
        for i in tqdm.tqdm(indices):
            current_number +=1
            orig_item, preproc_item = spider_data[i], val_data[i]

            with torch.no_grad():
                decoded = self._infer_one(self.model, orig_item, preproc_item, beam_size, output_history,
                                          use_heuristic)
                infer_output.write(
                    json.dumps({
                        'index': int(i),
                        'beams': decoded,
                    }) + '\n')
                infer_output.flush()

            if take_grad_steps:
                if batch_size =="1":
                    current_batch = [preproc_item]
                elif batch_size =="32":
                    if current_number %32 !=0:
                        current_batch.append(preproc_item)
                        clear_batch = False
                        continue
                    else:
                        clear_batch = True
                else:
                    current_batch.append(preproc_item)
                try:
                    with self.model_random:

                        loss = self.model.compute_loss(current_batch)
                        norm_loss = loss / self.finetune_config.num_batch_accumulated
                        norm_loss.backward()

                        if self.finetune_config.clip_grad:
                            torch.nn.utils.clip_grad_norm_(optimizer.bert_param_group["params"], \
                                                           self.finetune_config.clip_grad)
                        optimizer.step()
                        lr_scheduler.update_lr(last_step)
                        optimizer.zero_grad()
                    if clear_batch:
                        current_batch = []

                # stats = self._eval_model(self.logger, self.model, last_step, batch, 'val',
                #                          self.finetune_config.report_every_n)
                # val_losses.append(stats['loss'])
                except KeyError:
                    self.logger.log("keyError")
                    current_batch = []
                    continue
            # except AssertionError:
            #     self.logger.log("AssertionError")
            #     continue
        inferred = open(current_infer_output_path)
        metrics = spider_data.Metrics(spider_data)
        inferred_lines = list(inferred)
        # if len(inferred_lines) < len(spider_data):
        #     raise Exception(f'Not enough inferred: {len(inferred_lines)} vs {len(spider_data)}')

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
        #print(final_metrics['total_scores']['all']['exact'])
        scores.append((database, final_metrics['total_scores']['all']['exact'], len(indices)))
        # if last_step % self.finetune_config.save_every_n == 0:
        # saver.save(model_save_dir+'/seed_'+seed, last_step)

        #print('scores', scores)
        #print("average score:", self.aggregate_score(scores))
        return scores
    def aggregate_score(self, scores):
        total_num = 0
        total_score = 0
        for _, score, num_datapoint in scores:
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

    if os.path.exists(infer_output_path):
        print(f'Output file {infer_output_path} already exists')
        sys.exit(1)
    os.makedirs(os.path.dirname(infer_output_path), exist_ok=True)
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