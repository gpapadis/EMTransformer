import csv
import logging
import os
import resource
import time

import fcntl
import torch
from tqdm import tqdm, trange

from logging_customized import setup_logging
from model import save_model
from tensorboardX import SummaryWriter

setup_logging()


def train(device,
          train_dataloader,
          model,
          optimizer,
          scheduler,
          evaluation,
          num_epocs,
          max_grad_norm,
          save_model_after_epoch,
          experiment_name,
          output_dir,
          model_type):
    logging.info("***** Run training *****")
    tb_writer = SummaryWriter(os.path.join(output_dir, experiment_name))

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    # we are interested in 0 shot learning, therefore we already evaluate before training.
    eval_results = evaluation.evaluate(model, device, -1)
    for key, value in eval_results.items():
        tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

    train_time = 0
    for epoch in trange(int(num_epocs), desc="Epoch"):

        train_start = time.time()
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            model.train()

            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}

            if model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            tr_loss += loss.item()

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

            global_step += 1

            tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
            tb_writer.add_scalar('loss', (tr_loss - logging_loss), global_step)
            logging_loss = tr_loss
        train_time += time.time() - train_start
        train_max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        test_start = time.time()
        eval_results = evaluation.evaluate(model, device, epoch)
        test_time = time.time() - test_start
        test_max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        for key, value in eval_results.items():
            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

        if save_model_after_epoch:
            save_model(model, experiment_name, output_dir, epoch=epoch)

        if epoch == num_epocs - 1:
            result_file = '/home/remote/u6852937/projects/results.csv'
            file_exists = os.path.isfile(result_file)

            data_set_name = experiment_name.split('__')[0].lower()

            with open(result_file, 'a') as results_file:
                heading_list = ['method', 'dataset_name', 'train_time',
                                'test_time',
                                'train_max_mem', 'test_max_mem', 'TP', 'FP',
                                'FN',
                                'TN', 'Pre', 'Re', 'F1', 'Fstar']
                writer = csv.DictWriter(results_file, fieldnames=heading_list)

                if not file_exists:
                    writer.writeheader()

                fcntl.flock(results_file, fcntl.LOCK_EX)
                result_dict = {
                    'method': 'emtransformer-' + model_type,
                    'dataset_name': data_set_name,
                    'train_time': round(train_time, 2),
                    'test_time': round(test_time, 2),
                    'train_max_mem': train_max_mem,
                    'test_max_mem': test_max_mem,
                    'TP': eval_results['tp'],
                    'FP': eval_results['fp'],
                    'FN': eval_results['fn'],
                    'TN': eval_results['tn'],
                    'Pre': eval_results['precision'],
                    'Re': eval_results['recall'],
                    'F1': eval_results['f1'],
                    'Fstar': eval_results['fstar']
                }
                writer.writerow(result_dict)
                fcntl.flock(results_file, fcntl.LOCK_UN)

    tb_writer.close()
