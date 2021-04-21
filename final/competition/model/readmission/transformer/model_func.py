import os
import copy
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from utils import get_cuda


def training_process(model, epoch, dataloaders: dict, save_model=None, test=False):
    """
    Main function to call for model training.

    Must have at least training & test dataloaders

    Arguments:
    -----
        model: instantiation of model to be trained
        epoch: total number of epochs to train
        dataloaders: at least training dataloader, test/val optional
        device: cpu or gpu
        metric: auc or accuracy (acc)

    Returns:
        tuple containing:
            (training metrics, val metrics, test_metrics, feature importances evaluated on test data)
    """
    criterion = (
        nn.CrossEntropyLoss().cuda()
        if model.device_type != "cpu"
        else nn.CrossEntropyLoss()
    )

    lr = 0.002
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    if model.device_type == "gpu" and torch.cuda.is_available:
        model = model.cuda()
        device = get_cuda()
        model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    print(f"device: {model.module.device_type}")

    # initialize lists for documenting training performance
    pre_val_metric = -float("inf")
    train_loss = []
    train_metric = []

    val_loss = []
    val_metric = []

    test_loss = []
    test_metric = []

    for cur_epoch in range(epoch):
        print("-" * 10 + "Epoch {}/{}".format(cur_epoch + 1, epoch) + "-" * 30)

        epoch_train_loss, epoch_train_metric = epoch_train(
            model, dataloaders["train"], optimizer, criterion, test
        )
        train_metric.append(epoch_train_metric)
        print(
            "epoch_train_loss:",
            np.mean(epoch_train_loss),
            "epoch train AUC:",
            np.mean(epoch_train_metric),
        )

        # validation also provided
        if "val" in dataloaders:
            epoch_val_loss, epoch_val_metric, val_results = epoch_val(
                model, dataloaders["val"], optimizer, criterion, test
            )

            val_loss.append(epoch_val_loss)
            val_metric.append(epoch_val_metric)

            print(
                "epoch_val_loss:",
                np.mean(epoch_val_loss),
                "epoch val AUC:",
                np.mean(epoch_val_metric),
            )

            if epoch_val_metric > pre_val_metric:
                print("\t***Updating best model***")
                pre_val_metric = epoch_val_metric.copy()
                best_val_results = val_results

                if save_model is not None:
                    model_name = (
                        "emsize-{}_head-{}_layers-{}_epoch-{}_valauc-{}.pth".format(
                            model.module.emsize,
                            model.module.nhead,
                            model.module.nlayers,
                            str(epoch),
                            np.round(epoch_val_metric, decimals=3),
                        )
                    )

                    model_fp = os.path.join(save_model, model_name)
                    if not os.path.isdir(save_model):
                        os.makedirs(save_model)
                    torch.save(model.module.state_dict(), model_fp)

        if "test" in dataloaders:
            # predictions on test data
            epoch_test_loss, epoch_test_metric, test_results = epoch_val(
                model, dataloaders["test"], optimizer, criterion, test
            )
            test_loss.append(epoch_test_loss)
            test_metric.append(epoch_test_metric)
            print(
                "epoch_test_loss:",
                np.mean(epoch_test_loss),
                "epoch test AUC:",
                np.mean(epoch_test_metric),
            )

            # update best scores
            if "val" in dataloaders and epoch_val_metric > pre_val_metric:
                best_test_results = test_results

        scheduler.step()

    results = {"train_metric": train_metric}
    if len(val_metric) > 0:
        results["val_metric"] = val_metric
        results["val_results"] = best_val_results

    if len(test_metric) > 0:
        results["test_metric"] = test_metric
        results["test_results"] = best_test_results

    return results


def epoch_val(model, dataloader, optimizer, criterion, test):
    """
    Evaluate model performance, called by ModelProcess function

    Returns predictions, metrics and importance scores
    """
    epoch_loss = 0
    epoch_metric = 0

    model.eval()

    # initialize lists to compare predictions & ground truth labels
    # and extract importance scores for prediction
    all_ids = []
    order_labels = []
    prediction_scores = []
    events = []
    important_scores = []

    if test:
        counter = 0

    with torch.no_grad():
        for ids, seq, labels, mask in dataloader:
            # data formatting/loading
            labels = labels.squeeze().long()
            events.extend(seq.view(seq.size()[0], -1).squeeze().numpy())

            if (
                isinstance(model, nn.DataParallel) and model.module.device_type == "gpu"
            ) or (
                not isinstance(model, nn.DataParallel) and model.device_type == "gpu"
            ):
                seq, labels, mask = seq.cuda(), labels.cuda(), mask.cuda()

            predictions, importance = model(seq, mask=mask)

            loss = criterion(predictions, labels)

            important_scores.extend(importance.detach().cpu().numpy())
            order_labels.extend(labels.cpu().numpy())
            all_ids.extend(ids)
            prediction_scores.extend(
                F.softmax(predictions, dim=-1).detach().cpu().numpy()[:, 1]
            )

            epoch_loss += loss.item()

            epoch_metric = roc_auc_score(order_labels, prediction_scores)

            if test:
                counter += 1
                if counter >= 20:
                    break
    return (
        epoch_loss / len(dataloader),
        epoch_metric,
        [all_ids, order_labels, events, important_scores, prediction_scores],
    )


def epoch_train(model, dataloader, optimizer, criterion, test=0):
    """
    Model training, called by ModelProcess function

    Note: Does not return prediction importance scores
    """
    epoch_loss = 0
    epoch_metric = 0

    model.train()

    # initialize lists to compare predictions & ground truth labels
    # and extract importance scores for prediction
    order_labels = []
    prediction_scores = []

    if test:  # test function on a small number of batches
        counter = 0
    for ids, seq, labels, mask in dataloader:

        optimizer.zero_grad()

        labels = labels.squeeze().long()

        if (
            isinstance(model, nn.DataParallel) and model.module.device_type == "gpu"
        ) or (not isinstance(model, nn.DataParallel) and model.device_type == "gpu"):
            seq, labels, mask = seq.cuda(), labels.cuda(), mask.cuda()

        predictions, _ = model(seq, mask=mask)

        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        copied_labels = copy.deepcopy(labels.cpu().numpy())
        del labels
        order_labels.extend(copied_labels)

        copied_preds = copy.deepcopy(
            F.softmax(predictions, dim=-1).detach().cpu().numpy()[:, 1]
        )
        del predictions
        prediction_scores.extend(copied_preds)

        epoch_loss += loss.item()

        if test:
            counter += 1
            if counter >= test:
                break
    # calculate results
    epoch_metric = roc_auc_score(order_labels, prediction_scores)

    return epoch_loss / len(dataloader), epoch_metric
