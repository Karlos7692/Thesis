from datetime import datetime, timezone, timedelta


# TODO for now assume that its commsec asx closing time
def get_closing_time(date: str):
    dt = datetime.strptime(date, '%d %b %Y')
    return dt.replace(hour=14, minute=10, tzinfo=timezone(timedelta(hours=10)))
