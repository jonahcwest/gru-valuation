import sqlite3
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
