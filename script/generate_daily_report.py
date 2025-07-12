# scripts/generate_daily_report.py

import os
import django
from datetime import datetime, timedelta
from django.db.models import Count
from openai import OpenAI

# 设置 Django 环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from api.models import EventLog, Camera, DailyReport


def collect_data():
    today = datetime.now().date()
    start = datetime.combine(today, datetime.min.time())
    end = datetime.combine(today, datetime.max.time())

    events = EventLog.objects.filter(time__range=(start, end))

    summary = {
        '日期': str(today),
        '总事件数': events.count(),
        '未处理事件数': events.filter(status=0).count(),
        '处理中事件数': events.filter(status=1).count(),
        '已处理事件数': events.filter(status=2).count(),
    }

    type_counts = events.values('event_type').annotate(count=Count('event_type'))
    for item in type_counts:
        summary[f"类型:{item['event_type']}"] = item['count']

    total_cameras = Camera.objects.count()
    active_cameras = Camera.objects.filter(is_active=True).count()
    summary['摄像头总数'] = total_cameras
    summary['在线摄像头'] = active_cameras

    return summary


# === 步骤二：构建 Prompt（CoT）===
def build_prompt(summary):
    prompt = """
你是一个安防监控系统的智能助手，请根据以下数据生成一份简明的中文监控日报。
包括：总体事件概况、各类型事件情况、摄像头在线状态，以及必要时的风险提示。

下面是当天的监控数据摘要：
"""
    for k, v in summary.items():
        prompt += f"- {k}: {v}\n"
    prompt += """

请输出一段自然语言描述，总结当天监控情况。
"""
    return prompt


def generate_text_report():
    summary = collect_data()
    prompt = build_prompt(summary)

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是一个善于总结监控事件的助手。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5
    )

    content = response.choices[0].message.content

    # 写入数据库
    DailyReport.objects.update_or_create(date=datetime.now().date(), defaults={'content': content})
    print("✅ 日报已生成并写入数据库：\n", content)


if __name__ == '__main__':
    generate_text_report()
