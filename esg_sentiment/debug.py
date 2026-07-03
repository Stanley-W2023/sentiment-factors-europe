from data.db import Database
import pandas as pd
from config.constants import DB_PATH
with Database(DB_PATH) as db:
    data = db.readTable("STOXX")
    data.to_csv("current_stocks.csv")