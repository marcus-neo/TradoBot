import datetime
import pytz

tz = pytz.timezone("America/Cancun")
ct = datetime.datetime.now(tz=tz).date()
print(ct.isoformat())
timetest = datetime.datetime.fromisoformat('2007-10-10T00:00:00-04:00')
print(timetest.date().strftime("%Y-%m-%d"))

