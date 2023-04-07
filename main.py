# -*- coding:utf-8 -*-
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from data.datasets import input_dataset
from models import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument(
    "--noise_type",
    type=str,
    help="clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100",
    default="clean",
)
parser.add_argument(
    "--noise_path", type=str, help="path of CIFAR-10_human.pt", default=None
)
parser.add_argument(
    "--dataset", type=str, help=" cifar10 or cifar100", default="cifar10"
)
parser.add_argument("--n_epoch", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--print_freq", type=int, default=50)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    help="how many subprocesses to use for data loading",
)
parser.add_argument("--is_human", action="store_true", default=False)
parser.add_argument("--my_store", type=str, required=True)

args = parser.parse_args()

if args.noise_path is None:
    if args.dataset == "cifar10":
        args.noise_path = "./data/CIFAR-10_human.pt"
    elif args.dataset == "cifar100":
        args.noise_path = "./data/CIFAR-100_human.pt"
    else:
        raise NameError(f"Undefined dataset {args.dataset}")

noise_type_map = {
    "clean": "clean_label",
    "worst": "worse_label",
    "aggre": "aggre_label",
    "rand1": "random_label1",
    "rand2": "random_label2",
    "rand3": "random_label3",
    "clean100": "clean_label",
    "noisy100": "noisy_label",
}
args.noise_type = noise_type_map[args.noise_type]


################################################################################

import json
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

mp.set_sharing_strategy("file_system")

if torch.cuda.is_available():
    my_device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    my_device = torch.device("cpu")

my_ts = time.time_ns()
my_root = f"store/{args.my_store}/{my_ts}"
Path(my_root).mkdir(parents=True)
print(f"tail -f {my_root}/stderr.txt {my_root}/stdout.txt")

sys.stdout = open(f"{my_root}/stdout.txt", "w")
sys.stderr = open(f"{my_root}/stderr.txt", "w")

print(f"{my_root=}")

with open(f"{my_root}/args.json", "w") as f:
    json.dump(vars(args), f)

my_eval_datasets = {}
my_eval_datasets["train"], my_eval_datasets["test"], _, _ = input_dataset(
    args.dataset, args.noise_type, args.noise_path, args.is_human, my_eval=True
)

my_eval_loaders = {
    split: DataLoader(
        my_eval_datasets[split],
        batch_size=256,
        shuffle=False,
        num_workers=16,
        persistent_workers=True,
        pin_memory=True,
    )
    for split in ["train", "test"]
}

Path(f"{my_root}/epochs").mkdir()


def my_eval(net, opt, epoch):
    print()
    print("-" * 80)

    metrics = {
        "epochs": epoch + 1,
        "lr": opt.param_groups[0]["lr"],
    }

    Path(f"{my_root}/epochs/{epoch}").mkdir()
    net.eval()
    for split, eval_loader in my_eval_loaders.items():
        with torch.no_grad():
            yhats = []
            ys = []
            for x, y, _ in eval_loader:
                x = x.to(my_device, non_blocking=True)
                y = y.to(my_device, non_blocking=True)
                ys.append(y)
                yhat = net(x)
                yhats.append(yhat)
            yhat = torch.cat(yhats, dim=0)
            y = torch.cat(ys, dim=0)

            metrics.update(
                {
                    f"err/{split}": (yhat.argmax(-1) != y).float().mean().item(),
                    f"nll/{split}": F.cross_entropy(yhat, y).item(),
                }
            )

        torch.save(yhat, f"{my_root}/epochs/{epoch}/{split}_yhat.pt")
        print(f"saved {my_root}/epochs/{epoch}/{split}_yhat.pt")
        if epoch == 0:
            torch.save(y, f"{my_root}/epochs/{epoch}/{split}_y.pt")
            print(f"saved {my_root}/epochs/{epoch}/{split}_y.pt")

    with open(f"{my_root}/epochs/{epoch}/metrics.json", "w") as f:
        json.dump(metrics, f)
    print(f"saved {my_root}/epochs/{epoch}/metrics.json")

    print(json.dumps(metrics, indent=4))

    print("-" * 80)
    print()


################################################################################


# Adjust learning rate and for SGD Optimizer
def adjust_learning_rate(optimizer, epoch, alpha_plan):
    for param_group in optimizer.param_groups:
        param_group["lr"] = alpha_plan[epoch]


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# Train the Model
def train(epoch, train_loader, model, optimizer):
    train_total = 0
    train_correct = 0

    for i, (images, labels, indexes) in enumerate(train_loader):
        ind = indexes.cpu().numpy().transpose()
        batch_size = len(ind)

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        logits = model(images)

        prec, _ = accuracy(logits, labels, topk=(1, 5))
        # prec = 0.0
        train_total += 1
        train_correct += prec
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % args.print_freq == 0:
            print(
                "Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f"
                % (
                    epoch + 1,
                    args.n_epoch,
                    i + 1,
                    len(train_dataset) // batch_size,
                    prec,
                    loss.data,
                )
            )

    train_acc = float(train_correct) / float(train_total)
    return train_acc


# Evaluate the Model
def evaluate(test_loader, model):
    model.eval()  # Change model to 'eval' mode.
    correct = 0
    total = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()
    acc = 100 * float(correct) / float(total)

    return acc


#####################################main code ################################################


# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size = 128
learning_rate = args.lr

# load dataset
train_dataset, test_dataset, num_classes, num_training_samples = input_dataset(
    args.dataset, args.noise_type, args.noise_path, args.is_human
)
noise_prior = train_dataset.noise_prior
noise_or_not = train_dataset.noise_or_not
print("train_labels:", len(train_dataset.train_labels), train_dataset.train_labels[:10])

# load model
print("building model...")
model = ResNet34(num_classes)
print("building model done")
optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=128, num_workers=args.num_workers, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=64, num_workers=args.num_workers, shuffle=False
)
alpha_plan = [0.1] * 60 + [0.01] * 40
model.cuda()

epoch = 0
train_acc = 0

# training
noise_prior_cur = noise_prior
for epoch in range(args.n_epoch):
    # train models
    print(f"epoch {epoch}")
    adjust_learning_rate(optimizer, epoch, alpha_plan)
    model.train()
    train_acc = train(epoch, train_loader, model, optimizer)
    # evaluate models
    test_acc = evaluate(test_loader=test_loader, model=model)
    # save results
    print("train acc on train images is ", train_acc)
    print("test acc on test images is ", test_acc)

    my_eval(model, optimizer, epoch)
