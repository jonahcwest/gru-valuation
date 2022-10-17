from argparse import ArgumentParser
from datetime import datetime
import math
import os
import sqlite3
import time
import torch


DB_FILE = "fundamentals.db"
MODEL_STATE = "model.pt"
OPTIMIZER_STATE = "optimizer.pt"

COLUMNS = [
    "revenue",
    "ebit",
    "netinc",
    "ncfo",
    "fcf",
    "assets",
    "liabilities",
]


class Model(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super(Model, self).__init__()

        self.one = torch.nn.GRU(len(COLUMNS), hidden_size, batch_first=True)
        self.two = torch.nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor):
        x = self.one(x)[0]
        x = self.two(x)
        return x


class Data(torch.utils.data.Dataset):
    @staticmethod
    def company_to_tensors(company):
        financials = []

        for period in company:
            try:
                financials.append(
                    [
                        float(period[2]),
                        *[
                            float(period[i + 3]) / float(period[1])
                            for i in range(len(COLUMNS))
                        ],
                    ]
                )
            except ValueError:
                break

        if len(financials) < 2:
            return None

        financials.reverse()

        return scale_down(torch.tensor(financials))

    def __init__(self):
        con = sqlite3.connect(DB_FILE)
        cur = con.cursor()

        result = cur.execute(
            f"select ticker, fxusd, marketcap, {','.join(COLUMNS)} "
            "from data where dimension = 'MRQ' order by ticker, calendardate desc"
        )
        rows = result.fetchall()

        companies = []
        last = 0
        for row in rows:
            if row[0] != last:
                companies.append([])
                last = row[0]
            companies[-1].append(row)

        self.data = []
        for v in companies:
            tensors = Data.company_to_tensors(v)
            if tensors is None:
                continue
            self.data.append(tensors)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int):
        return self.data[i]


def try_to_float(x):
    try:
        return float(x)
    except ValueError:
        return 0


def scale_down(x: torch.Tensor):
    return (x.abs() + 1).log() * x.sign()


def scale_up(x: torch.Tensor):
    return (x.abs().exp() - 1) * x.sign()


def collate_fn(data):
    for i, x in enumerate(data):
        data[i] = torch.flip(x, [0])
    result = x = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
    return torch.flip(result, [1])


def try_lr(model, loss_fn, optimizer, data_loader, factor=1.05, start_lr=1e-5):
    lrs = [start_lr]
    losses = []
    best_loss = float("inf")
    data_iter = iter(data_loader)

    while (not len(losses)) or losses[-1] < best_loss * 3:
        for g in optimizer.param_groups:
            g["lr"] = lrs[-1]

        for param in model.parameters():
            param.grad = None

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
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()

        lrs.append(lrs[-1] * factor)

    return lrs[:-1], losses


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
    print("data loaded")

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
