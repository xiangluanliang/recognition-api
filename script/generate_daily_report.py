# scripts/generate_daily_report.py
import os
import django
from datetime import datetime
import pandas as pd
from django.db.models import Count
import openai  # 如果你用 openai api
from dotenv import load_dotenv

# 设置 Django 环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()
load_dotenv()

from api.models import EventLog, Camera


def format_summary(events):
    summary = {}
    summary['总事件数'] = events.count()
    summary['未处理'] = events.filter(status=0).count()
    summary['处理中'] = events.filter(status=1).count()
    summary['已处理'] = events.filter(status=2).count()

    type_counts = events.values('event_type').annotate(count=Count('event_type'))
    summary['事件类型统计'] = {item['event_type']: item['count'] for item in type_counts}

    cam_counts = events.values('camera__name').annotate(count=Count('camera')).order_by('-count')
    summary['高频摄像头'] = cam_counts[:3]

    serious_events = events.filter(event_type__in=['fire', 'conflict'])
    summary['严重事件'] = [{
        '时间': e.time.strftime('%H:%M'),
        '类型': e.get_event_type_display(),
        '摄像头': e.camera.name if e.camera else '未知'
    } for e in serious_events]

    return summary


def generate_ai_summary(event_summary: dict):
    lines = []
    lines.append(f"总事件数：{event_summary['总事件数']}")
    lines.append(f"未处理事件：{event_summary['未处理']}，处理中：{event_summary['处理中']}，已处理：{event_summary['已处理']}")

    lines.append("事件类型分布：")
    for k, v in event_summary['事件类型统计'].items():
        lines.append(f"  - {k}: {v} 起")

    lines.append("告警频繁的摄像头：")
    for cam in event_summary['高频摄像头']:
        lines.append(f"  - {cam['camera__name']}：{cam['count']} 起")

    if event_summary['严重事件']:
        lines.append("⚠️ 严重事件（需重点关注）：")
        for e in event_summary['严重事件']:
            lines.append(f"  - {e['时间']} 发生 {e['类型']}，来自摄像头 {e['摄像头']}")

    return "\n".join(lines)


def generate_report():
    today = datetime.now().date()
    start = datetime.combine(today, datetime.min.time())
    end = datetime.combine(today, datetime.max.time())
    events = EventLog.objects.filter(time__range=(start, end))

    event_summary = format_summary(events)
    prompt_text = f"""
你是一位负责车站安全的日报撰写员。请根据以下事件日志摘要撰写一篇监控日报：

{generate_ai_summary(event_summary)}

日报请用中文，清晰、正式地表达，开头写“今日安全监控日报”。
"""

    report_text = call_openai(prompt_text)

    # 存入数据库，供前端展示
    from api.models import DailyReport
    DailyReport.objects.create(date=today, content=report_text)

    print("📄 日报生成并保存至数据库：\n")
    print(report_text)


if __name__ == '__main__':
    generate_report()
