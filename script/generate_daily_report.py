# scripts/generate_daily_report.py

import os
import django
from datetime import datetime, timedelta
from django.db.models import Count
from django.utils import timezone

import sys
# print(sys.path)

# 设置 Django 环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.test')
django.setup()

from api.models import EventLog, Camera, DailyReport, AlarmLog

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 替换模型加载部分：
model_name = "Qwen/Qwen1.5-0.5B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).eval()

model = model.eval()


def collect_data():
    today = timezone.localdate()
    start = timezone.make_aware(datetime.combine(today, datetime.min.time()))
    end = timezone.make_aware(datetime.combine(today, datetime.max.time()))

    alarms = AlarmLog.objects.filter(time__range=(start, end))

    summary = {}

    # 日期信息
    summary['日期'] = str(today)

    # 总体告警情况
    summary['总事件数'] = alarms.count()
    summary['未处理事件数'] = alarms.filter(status=0).count()
    summary['处理中事件数'] = alarms.filter(status=1).count()
    summary['已处理事件数'] = alarms.filter(status=2).count()

    # 告警类型统计（用中文）
    EVENT_TYPE_MAP = dict(EventLog.EVENT_TYPE_CHOICES)
    type_counts = alarms.values('event__event_type').annotate(count=Count('id'))
    type_summary = {}
    for item in type_counts:
        etype = item['event__event_type']
        cname = EVENT_TYPE_MAP.get(etype, etype)
        type_summary[cname] = item['count']
    summary['事件类型统计'] = type_summary

    # 摄像头情况
    total_cameras = Camera.objects.count()
    active_cameras = Camera.objects.filter(is_active=True).count()
    summary['摄像头总数'] = total_cameras
    summary['在线摄像头'] = active_cameras
    summary['离线摄像头'] = total_cameras - active_cameras

    return summary

    # summary = {
    #     '日期': '2025-07-13',
    #     '总事件数': 14,
    #     '未处理事件数': 3,
    #     '处理中事件数': 4,
    #     '已处理事件数': 7,
    #     '类型:face_match': 5,
    #     '类型:fire': 2,
    #     '类型:intrusion': 6,
    #     '类型:conflict': 1,
    #     '摄像头总数': 20,
    #     '在线摄像头': 18,
    # }


def build_prompt(summary):
    prompt = (
        "你是一个安防监控系统的智能助手，请根据以下监控统计数据，撰写一份约 300 字的中文安防监控日报。"
        "内容应包括以下四部分（使用小标题分段）：①事件总体情况，②事件类型分布，③摄像头状态，④风险提示建议。\n"
        "注意：不要编造我未提供的信息，不要做过多主观猜测。\n"
        "特别说明：'区域入侵' 是指检测到人员进入了不允许进入的安全区域。\n\n"
    )

    prompt += f"📅 日期：{summary['日期']}\n"
    prompt += f"📊 总报警事件数：{summary['总事件数']}\n"
    prompt += f"🔴 未处理：{summary['未处理事件数']}，🟠 处理中：{summary['处理中事件数']}，🟢 已处理：{summary['已处理事件数']}\n\n"

    prompt += "📌 事件类型统计：\n"
    for k, v in summary['事件类型统计'].items():
        prompt += f"- {k}：{v} 起\n"

    prompt += "\n🎥 摄像头状态：\n"
    prompt += f"- 总数：{summary['摄像头总数']}，在线：{summary['在线摄像头']}，离线：{summary['离线摄像头']}\n"

    prompt += (
        "\n🛡️ 请使用清晰的小标题，输出一段 300 字的自然语言中文报告，总结上述监控情况。\n"
        "不要提及提示词、我提供的数据之外的内容，也不要使用假设或臆测语气。"
    )

    return prompt.strip()


#     prompt = """
# 你是一个安防监控系统的智能助手，请根据以下数据生成一份简明的中文监控日报。
# 包括：总体事件概况、各类型事件情况、摄像头在线状态，以及必要时的风险提示。
#
# 下面是当天的监控数据摘要：
# """
#     for k, v in summary.items():
#         prompt += f"- {k}: {v}\n"
#     prompt += """
#
# 请输出一段自然语言描述，总结当天监控情况。
# """
#     return prompt.strip()


def generate_text_report():
    summary = collect_data()
    prompt = build_prompt(summary)

    print("\n=== 📥 Prompt 输入 ===\n")
    print(prompt)

    messages = [
        {"role": "system", "content": "你是一个安防监控系统的智能助手"},
        {"role": "user", "content": prompt}
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # 模型推理
    outputs = model.generate(**inputs, max_new_tokens=512)

    # 解码模型输出部分（去掉输入内容）
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

    print("\n=== 📤 模型输出（生成的日报） ===\n")
    print(response)
    # 写入数据库
    # DailyReport.objects.update_or_create(date=timezone.localdate(), defaults={'content': content})
    # print("✅ 日报已生成并写入数据库：\n", content)


if __name__ == '__main__':
    print("\n=== 🧪 collect_data() 返回结果 ===\n")
    from pprint import pprint
    pprint(collect_data())

    generate_text_report()
