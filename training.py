import argparse
import os
import random

try:
    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from create_data import create
from models1 import GINConvNet
from utils import ci, collate, mse, pearson, rm2, rmse, spearman


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, device, train_loader, optimizer, loss_fn, epoch, log_interval):
    model.train()
    total_loss = 0.0
    total_samples = 0
    print(f"Training on {len(train_loader.dataset)} samples...")

    for batch_idx, data in enumerate(train_loader):
        data_mol = data[0].to(device)
        data_frags = data[1].to(device)
        data_pro = data[2].to(device)
        labels = data_mol.y.view(-1, 1).float().to(device)

        optimizer.zero_grad()
        output = model(data_mol, data_frags, data_pro)
        if isinstance(output, tuple):
            pred = output[0]
            aux_loss = output[1] if len(output) > 1 and torch.is_tensor(output[1]) else 0.0
        else:
            pred = output
            aux_loss = 0.0

        loss = loss_fn(pred, labels) + aux_loss
        loss.backward()
        optimizer.step()

        batch_samples = labels.size(0)
        total_loss += loss.item() * batch_samples
        total_samples += batch_samples

        if batch_idx % log_interval == 0:
            print(
                f"Train epoch: {epoch} [{batch_idx * batch_samples}/{len(train_loader.dataset)} "
                f"({100.0 * batch_idx / max(len(train_loader), 1):.0f}%)]\tLoss: {loss.item():.6f}"
            )

    return total_loss / max(total_samples, 1)


def predicting(model, device, loader):
    model.eval()
    total_preds = []
    total_labels = []
    print(f"Make prediction for {len(loader.dataset)} samples...")

    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device)
            data_frags = data[1].to(device)
            data_pro = data[2].to(device)
            output = model(data_mol, data_frags, data_pro)
            pred = output[0] if isinstance(output, tuple) else output
            total_preds.append(pred.cpu())
            total_labels.append(data_mol.y.view(-1, 1).cpu())

    return (
        torch.cat(total_labels, dim=0).numpy().flatten(),
        torch.cat(total_preds, dim=0).numpy().flatten(),
    )


def run_fold(args, fold):
    set_seed(args.seed + fold)
    dataset_index = {"kiba": 0, "davis": 1}[args.dataset]
    dataset_name = args.dataset

    print(f"\nRunning GINConvNet on {dataset_name}, fold {fold}")
    train_data, test_data = create(
        dataset_index,
        fold_index=fold,
        n_folds=args.n_folds,
        seed=args.seed,
        split_mode=args.split_mode,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = GINConvNet().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = f"GINConvNet_{dataset_name}_fold{fold}_seed{args.seed}"
    model_file = os.path.join(args.output_dir, f"model_{prefix}.pt")
    result_file = os.path.join(args.output_dir, f"result_{prefix}.csv")
    log_file = os.path.join(args.output_dir, f"log_{prefix}.csv")
    curve_file = os.path.join(args.output_dir, f"mse_curve_{prefix}.jpg")

    best_mse = float("inf")
    best_metrics = None
    best_epoch = -1
    curve_x = []
    curve_y = []

    with open(log_file, "w", encoding="utf-8") as log:
        log.write("epoch,train_loss,rmse,mse,pearson,spearman,ci,rm2,is_best\n")

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, device, train_loader, optimizer, loss_fn, epoch, args.log_interval)
            labels, preds = predicting(model, device, test_loader)
            metrics = [
                rmse(labels, preds),
                mse(labels, preds),
                pearson(labels, preds),
                spearman(labels, preds),
                ci(labels, preds),
                rm2(labels, preds),
            ]

            is_best = metrics[1] < best_mse
            if is_best:
                best_mse = metrics[1]
                best_metrics = metrics
                best_epoch = epoch
                torch.save(model.state_dict(), model_file)
                with open(result_file, "w", encoding="utf-8") as f:
                    f.write("rmse,mse,pearson,spearman,ci,rm2,best_epoch\n")
                    f.write(",".join([f"{x:.8f}" for x in metrics] + [str(best_epoch)]) + "\n")
                print(
                    f"Improved at epoch {best_epoch}: "
                    f"MSE={metrics[1]:.6f}, CI={metrics[4]:.6f}, Rm2={metrics[5]:.6f}"
                )
            else:
                print(
                    f"No improvement since epoch {best_epoch}: "
                    f"current MSE={metrics[1]:.6f}, best MSE={best_mse:.6f}"
                )

            log.write(
                f"{epoch},{train_loss:.6f},{metrics[0]:.6f},{metrics[1]:.6f},"
                f"{metrics[2]:.6f},{metrics[3]:.6f},{metrics[4]:.6f},{metrics[5]:.6f},{is_best}\n"
            )
            log.flush()

            if plt is not None:
                curve_x.append(epoch)
                curve_y.append(metrics[1])
                plt.figure(figsize=(10, 6))
                plt.plot(curve_x, curve_y, "b*--", alpha=0.6, linewidth=1, label="MSE")
                plt.legend()
                plt.xlabel("Epoch")
                plt.ylabel("MSE")
                plt.tight_layout()
                plt.savefig(curve_file, dpi=300)
                plt.close()

    return fold, best_epoch, best_metrics


def main():
    parser = argparse.ArgumentParser(description="Train MLI-DTA with five-fold cross-validation.")
    parser.add_argument("--dataset", choices=["davis", "kiba"], required=True)
    parser.add_argument("--fold", type=int, default=0, help="Fold index, from 0 to n_folds-1.")
    parser.add_argument("--run-all-folds", action="store_true", help="Train all folds sequentially.")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--split-mode", choices=["kfold", "deepdta"], default="kfold")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--output-dir", default="runs_5fold")
    args = parser.parse_args()

    folds = range(args.n_folds) if args.run_all_folds else [args.fold]
    results = [run_fold(args, fold) for fold in folds]

    if args.run_all_folds:
        metrics = np.array([item[2] for item in results], dtype=float)
        summary_file = os.path.join(args.output_dir, f"summary_GINConvNet_{args.dataset}_seed{args.seed}.csv")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("metric,mean,std\n")
            for name, values in zip(["rmse", "mse", "pearson", "spearman", "ci", "rm2"], metrics.T):
                f.write(f"{name},{values.mean():.8f},{values.std(ddof=1):.8f}\n")
        print(f"\nFive-fold summary saved to {summary_file}")


if __name__ == "__main__":
    main()
