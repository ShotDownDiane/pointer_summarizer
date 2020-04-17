import os
import sys
import time
import argparse
import torch

sys.path.append(os.path.join(os.getcwd(), 'data_util'))

from data_util import config
from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util.utils import calc_avg_loss, logger
from train_util import get_input_from_batch, get_output_from_batch
from model import Model

if config.gpus is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        logger.warning("NO gpu found, falling back to cpu backend.")
        device = torch.device("cpu")
else:
    device = torch.device("cpu")


class Evaluate(object):
    def __init__(self, model_file_or_model, vocab=None):
        if vocab is None:
            self.vocab = Vocab(config.vocab_path, config.vocab_size)
        else:
            assert isinstance(vocab, Vocab)
            self.vocab = vocab
        self.batcher = Batcher(config.eval_data_path, self.vocab, mode='eval',
                               batch_size=config.batch_size, single_pass=True)
        time.sleep(15)
        if isinstance(model_file_or_model, str):
            self.model = Model(device, model_file_or_model, is_eval=True)
        elif isinstance(model_file_or_model, Model):
            self.model = model_file_or_model
        else:
            raise ValueError("Cannot build model from type %s" % type(model_file_or_model))

    def eval_one_batch(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, device)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, device)

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
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps_for_log)
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage

            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_step_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_step_losses / dec_lens_var
        loss = torch.mean(batch_avg_loss)
        return loss.item()

    def run_eval(self):
        avg_loss, iter = 0, 0
        batch = self.batcher.next_batch()
        while batch is not None:
            loss = self.eval_one_batch(batch)
            avg_loss = calc_avg_loss(loss, avg_loss)
            iter += 1
            batch = self.batcher.next_batch()
        return avg_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-m",
                        dest="model_file_path",
                        required=False,
                        default=None,
                        help="Saved model used to evaluation.")
    args = parser.parse_args()
    eval_processor = Evaluate(args.model_file_path)
    avg_loss = eval_processor.run_eval()
    logger.info("Evaluation on model %s finished, avg_loss: %f" % (args.model_file_path, avg_loss))
