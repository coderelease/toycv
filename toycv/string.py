from datetime import datetime

import pytz


def time_str(time=None, timezone=None, Ymd_sep="-", HMS_sep=":", date_time_sep=" "):
    # 获取当前时间
    if not time:
        time = datetime.now()

    if not timezone:
        timezone = pytz.timezone('Asia/Shanghai')

    # 设置时区为+8（东八区）
    time = time.astimezone(timezone)

    # 将当前时间转换为字符串
    time = time.strftime(f"%Y{Ymd_sep}%m{Ymd_sep}%d{date_time_sep}%H{HMS_sep}%M{HMS_sep}%S")

    return time
