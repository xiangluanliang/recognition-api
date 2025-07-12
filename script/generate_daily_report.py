# scripts/generate_daily_report.py

import os
import django
from datetime import datetime, timedelta
from django.db.models import Count
from django.utils import timezone

# 设置 Django 环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.test')
django.setup()

from api.models import EventLog, Camera, DailyReport, AlarmLog

# 替代 openai，用 transformers 本地模型
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 初始化 CPU 轻量模型
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")


def collect_data():
    today = timezone.localdate()
    start = timezone.make_aware(datetime.combine(today, datetime.min.time()))
    end = timezone.make_aware(datetime.combine(today, datetime.max.time()))

    alarms = AlarmLog.objects.filter(time__range=(start, end))

    summary = {
        '日期': str(today),
        '总事件数': alarms.count(),
        '未处理事件数': alarms.filter(status=0).count(),
        '处理中事件数': alarms.filter(status=1).count(),
        '已处理事件数': alarms.filter(status=2).count(),
    }

    type_counts = alarms.values('event__event_type').annotate(count=Count('id'))
    for item in type_counts:
        summary[f"类型:{item['event__event_type']}"] = item['count']

    total_cameras = Camera.objects.count()
    active_cameras = Camera.objects.filter(is_active=True).count()
    summary['摄像头总数'] = total_cameras
    summary['在线摄像头'] = active_cameras

    return summary


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
    return prompt.strip()


def generate_text_report():
    summary = collect_data()
    prompt = build_prompt(summary)

    # 使用本地轻量模型生成日报
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=256)
    content = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 写入数据库
    DailyReport.objects.update_or_create(date=timezone.localdate(), defaults={'content': content})
    print("✅ 日报已生成并写入数据库：\n", content)


if __name__ == '__main__':
    generate_text_report()
