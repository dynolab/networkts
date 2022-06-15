import math
from datetime import datetime
import numpy as np


def time(total_minutes):
    days = (math.floor(total_minutes / (24*60)) % 7)
    weeks = math.floor(total_minutes / (24*60*7))
    leftover_minutes = total_minutes % (24*60)
    hours = math.floor(leftover_minutes / 60)
    mins = total_minutes - (days*1440) - (hours*60) - (weeks*24*60*7)
    days += 1
    if days == 7:
        days = 0
    return datetime.strptime(f'2004-{weeks}-{days} {hours}:{mins}',
                             '%Y-%W-%w %H:%M')
