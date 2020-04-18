import os

root_dir = os.path.expanduser("~")

# train_data_path = os.path.join(root_dir, "ptr_nw/cnn-dailymail-master/finished_files/train.bin")
train_data_path = os.path.join(root_dir, "cnn-dailymail/finished_files/chunked/train_*")
eval_data_path = os.path.join(root_dir, "cnn-dailymail/finished_files/val.bin")
decode_data_path = os.path.join(root_dir, "cnn-dailymail/finished_files/test.bin")
vocab_path = os.path.join(root_dir, "cnn-dailymail/finished_files/vocab")
ouput_root = os.path.join(root_dir, "pointer_summarizer/output")

# Hyperparameters
hidden_dim = 256
emb_dim = 128
batch_size = 8
max_enc_steps = 400
max_dec_steps = 100
beam_size = 4
min_dec_steps = 35
vocab_size = 50000
lr = 0.15
adagrad_init_acc = 0.1
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4
max_grad_norm = 2.0
pointer_gen = True
is_coverage = True
cov_loss_wt = 1.0

eps_for_log = 1e-12
max_iterations = 500000
print_interval = 100
# if eval_interval is set, we only keep best model
eval_interval = 1000
patience = 10
# if eval_interval is not set, please specify save_interval
save_interval = 1000
random_seed = 2020

# gpu id for training, use None if not using gpu
gpus = "2,3"

lr_coverage = 0.15

forcing_ratio = 0.75  # initial percentage of using teacher forcing
forcing_decay_type = 'sigmoid'  # linear, exp (exponential), sig(inverse-sigmoid), or None
decay_to_0_iter = int(max_iterations / 3 * 2)  # change this according to convergence speed
