# config for construct_examplars
batch_size = 16 # 或者64
shuffle = True
train_examplar_size = 1000   # FIXME：为了测试改为20
# train_examplar_size = 20
valid_examplar_size = 100
# valid_examplar_size = 10
examplar_method_percentage = 0.7
return_selected_ids = False

# config for model.py
model_name_or_path = "/home/xg/AI-ModelScope/t5-base"
tokenizer_name_or_path = "/home/xg/AI-ModelScope/t5-base"

# config for run
data_dir = "/home/xg/life_long_NER2/ner_data/ontonotes/ontoNotes_standard/"
device = "cuda:1"
src_max_length = 256
tgt_max_length = 155
weight_decay = 0.0
learning_rate = 3e-4  # 比起3e-5更好
adam_epsilon = 1e-8
print_iters_eval = 50
experiment_type = "life_long"
adaptive = False   # TODO 我们在这里改为false了
ewc_importance = 90000  # 看情况改变
max_grad_norm = 1.0
print_iters = 10     # FIXME 这事为了debug，从10改为5
task_sequence = [
        "per",   # 2013
        "gpe", # 2016
        "org",   # 2017.9
        "date",
        "card",
        "norp"]
debug = False
train_epochs = 10
