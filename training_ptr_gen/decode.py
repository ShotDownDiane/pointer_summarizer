# Except for the pytorch part content of this file is copied from https://github.com/abisee/pointer-generator/blob/master/

import os
import sys
import time
import torch
import argparse

sys.path.append(os.path.join(os.getcwd(), 'data_util'))
from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util import data, config
from model import Model
from data_util.utils import write_for_rouge, rouge_eval, rouge_log, logger
from train_util import get_input_from_batch

if config.gpus is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        logger.warning("NO gpu found, falling back to cpu backend.")
        device = torch.device("cpu")
else:
    device = torch.device("cpu")


class Beam(object):
    def __init__(self, tokens, log_probs, state, context, coverage):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context
        self.coverage = coverage

    def extend(self, token, log_prob, state, context, coverage):
        return Beam(tokens=self.tokens + [token],
                    log_probs=self.log_probs + [log_prob],
                    state=state,
                    context=context,
                    coverage=coverage)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)


class BeamSearch(object):
    def __init__(self, model_file_path):
        model_name = os.path.basename(model_file_path)
        self._decode_dir = os.path.join(config.ouput_root, 'decode_%s' % model_name)
        self._rouge_ref_dir = os.path.join(self._decode_dir, 'rouge_ref')
        self._rouge_dec_dir = os.path.join(self._decode_dir, 'rouge_dec_dir')
        for p in [self._decode_dir, self._rouge_ref_dir, self._rouge_dec_dir]:
            if not os.path.exists(p):
                os.makedirs(p)

        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.decode_data_path, self.vocab, mode='decode',
                               batch_size=config.beam_size, single_pass=True)
        time.sleep(15)

        self.model = Model(device, model_file_path, is_eval=True)

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def decode(self):
        start = time.time()
        counter = 0
        batch = self.batcher.next_batch()
        while batch is not None:
            # Run beam search to get best Hypothesis
            best_summary = self.beam_search(batch)

            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            decoded_words = data.outputids2words(output_ids, self.vocab,
                                                 (batch.art_oovs[0] if config.pointer_gen else None))

            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            original_abstract_sents = batch.original_abstracts_sents[0]

            write_for_rouge(original_abstract_sents, decoded_words, counter,
                            self._rouge_ref_dir, self._rouge_dec_dir)
            counter += 1
            if counter % config.print_interval == 0:
                logger.info('%d example in %d sec' % (counter, time.time() - start))
                start = time.time()

            batch = self.batcher.next_batch()

        logger.info("Decoder has finished reading dataset for single_pass.")
        logger.info("Now starting ROUGE eval...")
        results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        rouge_log(results_dict, self._decode_dir)

    def beam_search(self, batch):
        # batch should have only one example
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0 = \
            get_input_from_batch(batch, device)

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_0 = self.model.reduce_state(encoder_hidden)

        dec_h, dec_c = s_t_0  # 1 x 2*hidden_size
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        # decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[self.vocab.word2id(data.START_DECODING)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context=c_t_0[0],
                      coverage=(coverage_t_0[0] if config.is_coverage else None))
                 for _ in range(config.beam_size)]
        results = []  # this will contain finished hypotheses (those that have emitted the [STOP] token)
        steps = 0
        while steps < config.max_dec_steps and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]  # latest token produced by each hypothesis
            # change any in-article temporary OOV ids to [UNK] id, so that we can lookup word embeddings
            latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(data.UNKNOWN_TOKEN) \
                             for t in latest_tokens]
            y_t_1 = torch.tensor(latest_tokens, dtype=torch.long).to(device)
            all_state_h = []
            all_state_c = []

            all_context = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)

                all_context.append(h.context)

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)

            coverage_t_1 = None
            if config.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)
            # Run one step of the decoder to get the new info
            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(y_t_1, s_t_1,
                                                                                    encoder_outputs, encoder_feature,
                                                                                    enc_padding_mask, c_t_1,
                                                                                    extra_zeros, enc_batch_extend_vocab,
                                                                                    coverage_t_1, steps)
            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()
            # Extend each hypothesis and collect them all in all_beams
            all_beams = []
            # On the first step, we only had one original hypothesis (the initial hypothesis).
            # On subsequent steps, all original hypotheses are distinct.
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                # take the ith hypothesis and new decoder state info
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if config.is_coverage else None)

                # for each of the top 2*beam_size hypothesis, extend the ith hypothesis with the jth option
                for j in range(config.beam_size * 2):
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                        log_prob=topk_log_probs[i, j].item(),
                                        state=state_i,
                                        context=context_i,
                                        coverage=coverage_i)
                    all_beams.append(new_beam)

            # Filter and collect any hypotheses that have produced the end token.
            beams = []  # will contain hypotheses for the next step
            for h in self.sort_beams(all_beams):  # in order of most likely h
                if h.latest_token == self.vocab.word2id(data.STOP_DECODING):  # if stop token is reached...
                    # If this hypothesis is sufficiently long, put in results. Otherwise discard.
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:  # hasn't reached stop token, so continue to extend this hypothesis
                    beams.append(h)
                # Once we've collected beam_size-many hypotheses for the next step,
                # or beam_size-many complete hypotheses, stop.
                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break

            steps += 1
        # At this point, either we've got beam_size results, or we've reached maximum decoder steps
        # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
        if len(results) == 0:
            results = beams

        # Sort hypotheses by average log probability
        beams_sorted = self.sort_beams(results)

        # Return the hypothesis with highest average log prob
        return beams_sorted[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-m",
                        dest="model_file_path",
                        required=False,
                        default=None,
                        help="Saved model used to test.")
    args = parser.parse_args()
    beam_Search_processor = BeamSearch(args.model_file_path)
    beam_Search_processor.decode()
