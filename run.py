import torch
import torch.nn as nn
import json
from trainer import train,eval
from cost import crit_weights_gen
from net import Net
from dataset import NerDataset, VOCAB, pad
import torch.optim as optim
import os

batch_size = 32
lr = 0.001
n_epochs = 20
finetuning = True
top_rnns = True
trainset = "data/Bangla-NER-Splitted-Dataset.json"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Net(top_rnns, len(VOCAB), device, finetuning)
model.to(device)

with open(trainset) as infile:
    data = json.load(infile)

new = data['train']
train_texts, train_labels = list(zip(*map(lambda d: (d['sentence'], d['iob_tags']), new)))
new = data['validation']
valid_texts, valid_labels = list(zip(*map(lambda d: (d['sentence'], d['iob_tags']), new)))
new = data['test']
test_texts, test_labels = list(zip(*map(lambda d: (d['sentence'], d['iob_tags']), new)))

sents_train, tags_li_train = [], []
for x in train_texts:
    sents_train.append(["[CLS]"] + x + ["[SEP]"])
for y in train_labels:
    tags_li_train.append(["<PAD>"] + y + ["<PAD>"])

sents_valid, tags_li_valid = [], []
for x in valid_texts:
    sents_valid.append(["[CLS]"] + x + ["[SEP]"])
for y in valid_labels:
    tags_li_valid.append(["<PAD>"] + y + ["<PAD>"])

sents_test, tags_li_test = [], []
for x in test_texts:
    sents_test.append(["[CLS]"] + x + ["[SEP]"])
for y in test_labels:
    tags_li_test.append(["<PAD>"] + y + ["<PAD>"])


train_dataset = NerDataset(sents_train, tags_li_train)
eval_dataset = NerDataset(sents_valid, tags_li_valid)
test_dataset = NerDataset(sents_test, tags_li_test)

train_iter = torch.utils.data.DataLoader(dataset=train_dataset,
                             batch_size= batch_size,
                             shuffle=True,
                             collate_fn=pad,
                             num_workers=0
                             )
eval_iter = torch.utils.data.DataLoader(dataset=eval_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn = pad,
                             num_workers=0
                             )
test_iter = torch.utils.data.DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn = pad,
                             num_workers=0
                             )

optimizer = optim.Adam(model.parameters(), lr = lr)
data_dist = [7237, 15684, 714867, 759, 20815, 9662, 8512, 37529, 70025]
crit_weights = crit_weights_gen(0.5,0.9,data_dist)
#insert 0 cost for ignoring <PAD>
crit_weights.insert(0,0)
crit_weights = torch.tensor(crit_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=crit_weights)


from focal_loss import FocalLoss
alpha=[1.0]*10
focal_loss_fn = FocalLoss(gamma=2, alpha=alpha, reduction = "mean", device=device)


macro_avg_f1 = []
best_macro_avg_f1 = -1
best_model_name = ''

# remove all models before starting, if any exist in models folder
os.system(f'rm -rf {os.path.join("models", "*")}')

for epoch in range(1, n_epochs+1):
    if epoch>5:
        optimizer = optim.Adam([
                                {"params": model.fc.parameters(), "lr": 0.0005},
                                {"params": model.xlmr.parameters(), "lr": 5e-5},
                                {"params": model.rnn.parameters(), "lr": 0.0005},
                                {"params": model.crf.parameters(), "lr": 0.0005}
                                ],)
    train(model, train_iter, optimizer, criterion, epoch, focal_loss_fn)
    _, dict_report = eval(model, eval_iter, epoch)
    current_macro_avg_f1 = dict_report['macro avg']['f1-score']
    macro_avg_f1.append(current_macro_avg_f1)
    fname = os.path.join("models", str(epoch))

    if(current_macro_avg_f1 > best_macro_avg_f1):
        best_macro_avg_f1 = current_macro_avg_f1
        best_model_index = f"{fname}.pt"
        # remove all models from models folder
        os.system(f'rm -rf {os.path.join("models", "*")}')
        torch.save(model.state_dict(), f"{fname}.pt")


print('===========================================')
print('Training completed. Evaluating on Test Set')
print('===========================================')

best_index = 0
for i in range(len(macro_avg_f1)):
    if(macro_avg_f1[i] > macro_avg_f1[best_index]):
        best_index = i

print(f'Best macro avg f1 on validation set: {macro_avg_f1[best_index]}, at epoch {best_index+1}')
print(f'Evaluating best model on test set:')
model.load_state_dict(torch.load(f'models/{best_index+1}.pt'))
_, dict_report = eval(model, test_iter, 'test set')
macro_avg_f1.append(dict_report['macro avg']['f1-score'])
with open(os.path.join('f1-macro-avg-all-epochs.txt'), 'w') as output_file:
    output_file.write(str(macro_avg_f1))

print(f'results: {macro_avg_f1}')


