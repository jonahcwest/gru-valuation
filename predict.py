from argparse import ArgumentParser
import sqlite3
import torch
from lib import COLUMNS, FUNDAMENTALS_DB, scale_up
from train import MODEL_STATE, Data, Model


def main():
    parser = ArgumentParser()
    parser.add_argument("ticker", type=str)
    parser.add_argument("--hidden-size", type=int)
    args = parser.parse_args()

    model = Model(args.hidden_size)
    model.load_state_dict(torch.load(MODEL_STATE))

    con = sqlite3.connect(FUNDAMENTALS_DB)
    cur = con.cursor()

    result = cur.execute(
        f"select ticker, fxusd, marketcap, {','.join(COLUMNS)} "
        "from data where dimension = 'MRQ' and ticker = ? order by ticker, calendardate desc",
        (args.ticker,),
    )
    rows = result.fetchall()
    input = Data.company_to_tensors(rows)

    output = model(input[:, 1:])
    output = scale_up(output)
    for v in output:
        print(" | ".join(map(lambda x: "{:>17,}".format(int(x.item())), v)))


main()
