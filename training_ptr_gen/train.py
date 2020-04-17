import os
import sys
import time
import argparse

import tensorflow as tf
import torch
from model import Model
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adagrad

sys.path.append(os.path.join(os.getcwd(), 'data_util'))

from data_util import config
from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util.utils import calc_avg_loss, logger
from train_util import get_input_from_batch, get_output_from_batch
from eval import Evaluate

if config.gpus is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        logger.warning("NO gpu found, falling back to cpu backend.")
        device = torch.device("cpu")
else:
    device = torch.device("cpu")


class Train(object):
    def __init__(self):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                               batch_size=config.batch_size, single_pass=False)
        time.sleep(15)

        train_dir = os.path.join(config.ouput_root, 'train_%d' % (int(time.time())))
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        self.checkpoint_dir = os.path.join(train_dir, 'checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.train_summary_writer = tf.summary.create_file_writer(os.path.join(train_dir, 'log', 'train'))
        self.eval_summary_writer = tf.summary.create_file_writer(os.path.join(train_dir, 'log', 'eval'))

    def save_model(self, model_path, running_avg_loss, iter):
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        torch.save(state, model_path)

    def setup_train(self, model_file_path=None):
        self.model = Model(device, model_file_path)

        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())
        initial_lr = config.lr_coverage if config.is_coverage else config.lr
        self.optimizer = Adagrad(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)

        start_iter, start_loss = 0, 0

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not config.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device)

        return start_iter, start_loss

    def train_one_batch(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, device)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, device)

        self.optimizer.zero_grad()

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden)

        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                                                           encoder_outputs,
                                                                                           encoder_feature,
                                                                                           enc_padding_mask, c_t_1,
                                                                                           extra_zeros,
                                                                                           enc_batch_extend_vocab,
                                                                                           coverage, di)
            # use elements in target as the indexes of dimension 1 of final_dist to gather the elements in final_dist
            # target:[batch_size,1], final_dist:[batch_size, vocab_size+batch_oov_size],
            # gold_probs has the same size as index=target, and gold_probs[i][j]=[i][target[i][j]]
            target = target_batch[:, di].unsqueeze(1)
            gold_probs = torch.gather(final_dist, dim=1, index=target)
            gold_probs = gold_probs.squeeze()
            step_loss = -torch.log(gold_probs + config.eps_for_log)
            if config.is_coverage:  # if not using coverge, keep coverage=None
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage

            step_mask = dec_padding_mask[:, di]  # padding in target should not count into loss
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses / dec_lens_var
        loss = torch.mean(batch_avg_loss)

        loss.backward()

        self.norm = clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()

        return loss.item()

    def train(self, n_iters, init_model_path=None):
        iter, avg_loss = self.setup_train(init_model_path)
        start = time.time()
        cnt = 0
        best_model_path = None
        min_eval_loss = float('inf')
        while iter < n_iters:
            batch = self.batcher.next_batch()
            loss = self.train_one_batch(batch)
            model_path = os.path.join(self.checkpoint_dir, 'model_step_%d' % (iter + 1))
            avg_loss = calc_avg_loss(loss, avg_loss)

            if (iter + 1) % config.print_interval == 0:
                with self.train_summary_writer.as_default():
                    tf.summary.scalar(name='loss', data=loss, step=iter)
                self.train_summary_writer.flush()
                logger.info(
                    'steps %d, took %.2f seconds, train avg loss: %f' % (iter + 1, time.time() - start, avg_loss))
                start = time.time()
            if config.eval_interval is not None and (iter + 1) % config.eval_interval == 0:
                start = time.time()
                logger.info("Start Evaluation on model %s" % model_path)
                eval_processor = Evaluate(self.model, self.vocab)
                eval_loss = eval_processor.run_eval()
                logger.info("Evaluation finished, took %.2f seconds, eval loss: %f" % (time.time() - start, eval_loss))
                with self.eval_summary_writer.as_default():
                    tf.summary.scalar(name='eval_loss', data=eval_loss, step=iter)
                self.eval_summary_writer.flush()
                if eval_loss < min_eval_loss:
                    logger.info("This is the best model so far, saving it to disk.")
                    min_eval_loss = eval_loss
                    best_model_path = model_path
                    self.save_model(model_path, eval_loss, iter)
                    cnt = 0
                else:
                    cnt += 1
                    if cnt > config.patience:
                        logger.info(
                            "Eval loss doesn't drop for %d straight times, early stopping.\n"
                            "Best model: %s (Eval loss %f: )" % (
                                config.patience, best_model_path, min_eval_loss))
                        break
                start = time.time()
            elif (iter + 1) % config.save_interval == 0:
                self.save_model(model_path, avg_loss, iter)
            iter += 1
        else:
            logger.info("Training finished, best model: %s, with train loss %f: " % (best_model_path, min_eval_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-m",
                        dest="model_file_path",
                        required=False,
                        default=None,
                        help="Resume training from saved model(default: None for training from scratch).")
    args = parser.parse_args()

    train_processor = Train()
    train_processor.train(config.max_iterations, args.model_file_path)
