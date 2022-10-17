from argparse import ArgumentParser
from datetime import datetime
import math
import os
import time
import torch
from lib import MODEL_STATE, OPTIMIZER_STATE, Data, Model


def collate_fn(data):
    for i, x in enumerate(data):
        data[i] = torch.flip(x, [0])
    result = x = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
    return torch.flip(result, [1])


def main():
    parser = ArgumentParser()
    parser.add_argument("--hidden-size", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--log-interval", required=True, type=int)
    parser.add_argument("--optimizer", required=True, type=str)
    parser.add_argument("--lr", required=True, type=float)
    parser.add_argument("--lr-time", required=True, type=float)
    parser.add_argument("--lr-target", required=True, type=float)
    args = parser.parse_args()

    data_loader = torch.utils.data.DataLoader(
        Data(),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_iter = iter(data_loader)

    model = Model(args.hidden_size)
    if torch.cuda.is_available():
        model.cuda()
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    if os.path.isfile(MODEL_STATE):
        model.load_state_dict(torch.load(MODEL_STATE))
    if os.path.isfile(OPTIMIZER_STATE):
        optimizer.load_state_dict(torch.load(OPTIMIZER_STATE))

    lr_factor = (args.lr_target / args.lr) ** (1 / (args.lr_time * 60 * 60))

    start_time = time.time()
    elapsed = start_time
    loss_sum = 0
    loss_sq_sum = 0
    sum_steps = 0

    while True:
        for x in optimizer.param_groups:
            x["lr"] = args.lr * lr_factor ** (time.time() - start_time)

        try:
            financials = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            continue
        if torch.cuda.is_available():
            financials = financials.cuda()

        for param in model.parameters():
            param.grad = None

        output = model(financials[:, :, 1:])
        loss = loss_fn(output, financials[:, :, :1])
        with torch.no_grad():
            loss_sum += loss
            loss_sq_sum += loss**2
        sum_steps += 1

        if loss.isnan():
            raise ValueError

        if time.time() - elapsed > args.log_interval:
            torch.save(model.state_dict(), MODEL_STATE)
            torch.save(optimizer.state_dict(), OPTIMIZER_STATE)

            std_dev = math.sqrt(loss_sq_sum / sum_steps - (loss_sum / sum_steps) ** 2)
            avg_loss = loss_sum.item() / sum_steps

            print(
                " | ".join(
                    map(
                        lambda x: "{:>10.8}".format(x),
                        [
                            datetime.now().strftime("%H:%M:%S"),
                            avg_loss,
                            std_dev / avg_loss * 100,
                            args.lr * lr_factor ** (time.time() - start_time),
                            sum_steps / args.log_interval,
                        ],
                    )
                )
            )

            elapsed = time.time()
            loss_sum = 0
            loss_sq_sum = 0
            sum_steps = 0

        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    main()
