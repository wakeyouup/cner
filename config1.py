# config for construct_examplars
batch_size = 16 # 或者64
shuffle = True
train_examplar_size = 1000
valid_examplar_size = 100
examplar_method_percentage = 1.0
return_selected_ids = False

# config for model.py
model_name_or_path = "/home/xg/AI-ModelScope/t5-base"
tokenizer_name_or_path = "/home/xg/AI-ModelScope/t5-base"

# config for run
data_dir = "/home/xg/life_long_NER2/ner_data/ontonotes/ontonote_few_test"
device = "cuda:0"
src_max_length = 256
tgt_max_length = 155
weight_decay = 0.0
learning_rate = 1e-4  # 比起3e-5更好, 再试试看1e-4
adam_epsilon = 1e-8
print_iters_eval = 50
experiment_type = "life_long"
adaptive = True
ewc_importance = 100000  # 看情况改变
max_grad_norm = 1.0
print_iters = 10
task_sequence = [
        "per",   # 2013
        "org", # 2016
        "gpe",
        "date",
        "card",
        "norp"
        ]
debug = False
train_epochs = 20

# use_examplar = False
