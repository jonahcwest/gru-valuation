import itertools
import os
import sqlite3

from financials.train import DB_FILE


DATA_FILE = "fundamentals.csv"

DATA_START = 1
COMMIT_INTERVAL = 300_000


def main():
    file = open(DATA_FILE)

    if os.path.isfile(DB_FILE):
        raise ValueError("db exists")
    con = sqlite3.connect("fundamentals.db")
    cur = con.cursor()

    columns = (
        ("ticker", "blob"),
        ("dimension", "blob"),
        ("calendardate", "blob"),
        ("datekey", "blob"),
        ("reportperiod", "blob"),
        ("lastupdated", "blob"),
        ("accoci", "integer"),
        ("assets", "integer"),
        ("assetsavg", "integer"),
        ("assetsc", "integer"),
        ("assetsnc", "integer"),
        ("assetturnover", "integer"),
        ("bvps", "integer"),
        ("capex", "integer"),
        ("cashneq", "integer"),
        ("cashnequsd", "integer"),
        ("cor", "integer"),
        ("consolinc", "integer"),
        ("currentratio", "integer"),
        ("de", "integer"),
        ("debt", "integer"),
        ("debtc", "integer"),
        ("debtnc", "integer"),
        ("debtusd", "integer"),
        ("deferredrev", "integer"),
        ("depamor", "integer"),
        ("deposits", "integer"),
        ("divyield", "integer"),
        ("dps", "integer"),
        ("ebit", "integer"),
        ("ebitda", "integer"),
        ("ebitdamargin", "integer"),
        ("ebitdausd", "integer"),
        ("ebitusd", "integer"),
        ("ebt", "integer"),
        ("eps", "integer"),
        ("epsdil", "integer"),
        ("epsusd", "integer"),
        ("equity", "integer"),
        ("equityavg", "integer"),
        ("equityusd", "integer"),
        ("ev", "integer"),
        ("evebit", "integer"),
        ("evebitda", "integer"),
        ("fcf", "integer"),
        ("fcfps", "integer"),
        ("fxusd", "integer"),
        ("gp", "integer"),
        ("grossmargin", "integer"),
        ("intangibles", "integer"),
        ("intexp", "integer"),
        ("invcap", "integer"),
        ("invcapavg", "integer"),
        ("inventory", "integer"),
        ("investments", "integer"),
        ("investmentsc", "integer"),
        ("investmentsnc", "integer"),
        ("liabilities", "integer"),
        ("liabilitiesc", "integer"),
        ("liabilitiesnc", "integer"),
        ("marketcap", "integer"),
        ("ncf", "integer"),
        ("ncfbus", "integer"),
        ("ncfcommon", "integer"),
        ("ncfdebt", "integer"),
        ("ncfdiv", "integer"),
        ("ncff", "integer"),
        ("ncfi", "integer"),
        ("ncfinv", "integer"),
        ("ncfo", "integer"),
        ("ncfx", "integer"),
        ("netinc", "integer"),
        ("netinccmn", "integer"),
        ("netinccmnusd", "integer"),
        ("netincdis", "integer"),
        ("netincnci", "integer"),
        ("netmargin", "integer"),
        ("opex", "integer"),
        ("opinc", "integer"),
        ("payables", "integer"),
        ("payoutratio", "integer"),
        ("pb", "integer"),
        ("pe", "integer"),
        ("pe1", "integer"),
        ("ppnenet", "integer"),
        ("prefdivis", "integer"),
        ("price", "integer"),
        ("ps", "integer"),
        ("ps1", "integer"),
        ("receivables", "integer"),
        ("retearn", "integer"),
        ("revenue", "integer"),
        ("revenueusd", "integer"),
        ("rnd", "integer"),
        ("roa", "integer"),
        ("roe", "integer"),
        ("roic", "integer"),
        ("ros", "integer"),
        ("sbcomp", "integer"),
        ("sgna", "integer"),
        ("sharefactor", "integer"),
        ("sharesbas", "integer"),
        ("shareswa", "integer"),
        ("shareswadil", "integer"),
        ("sps", "integer"),
        ("tangibles", "integer"),
        ("taxassets", "integer"),
        ("taxexp", "integer"),
        ("taxliabilities", "integer"),
        ("tbvps", "integer"),
        ("workingcapital", "integer"),
    )

    statement = (
        f"create table data ({','.join(map(lambda x: f'{x[0]} {x[1]}', columns))}, "
        "primary key (ticker, dimension, calendardate, lastupdated, reportperiod, datekey))"
    )
    cur.execute(statement)

    for i, line in enumerate(itertools.islice(file, DATA_START, None)):
        split = line.split(",")
        statement = f"insert into data ({','.join(map(lambda x: x[0], columns))}) values ({','.join(['?'] * len(split))})"
        cur.execute(statement, split)

        if i % COMMIT_INTERVAL == COMMIT_INTERVAL - 1:
            con.commit()
            print("commit at", i)

    con.commit()


main()
