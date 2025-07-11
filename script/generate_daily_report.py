# scripts/generate_daily_report.py
import os
import django
from datetime import datetime, timedelta
import pandas as pd

# 设置 Django 环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from api.models import EventLog, Camera


def generate_report():
    today = datetime.now().date()
    start = datetime.combine(today, datetime.min.time())
    end = datetime.combine(today, datetime.max.time())

    events = EventLog.objects.filter(time__range=(start, end))

    summary = {
        '日期': str(today),
        '总事件数': events.count(),
        '未处理事件数': events.filter(status=0).count(),
        '处理中的事件数': events.filter(status=1).count(),
        '已处理事件数': events.filter(status=2).count(),
    }

    # 各类型事件统计
    type_counts = events.values('event_type').annotate(count=pd.Count('event_type'))
    for item in type_counts:
        summary[f"事件类型-{item['event_type']}"] = item['count']

    # 摄像头状态
    total_cameras = Camera.objects.count()
    active_cameras = Camera.objects.filter(is_active=True).count()
    summary['摄像头总数'] = total_cameras
    summary['在线摄像头'] = active_cameras

    # 创建 DataFrame 表格
    detail_data = [{
        '时间': e.time.strftime('%Y-%m-%d %H:%M:%S'),
        '类型': e.get_event_type_display(),
        '状态': e.get_status_display(),
        '摄像头': str(e.camera),
        '截图': e.image_path,
        '视频': e.video_clip_path,
    } for e in events]

    df_detail = pd.DataFrame(detail_data)
    df_summary = pd.DataFrame([summary])

    # 保存日报
    report_dir = '/root/autodl-tmp/reports'
    os.makedirs(report_dir, exist_ok=True)

    file_path = os.path.join(report_dir, f'report_{today}.xlsx')
    with pd.ExcelWriter(file_path) as writer:
        df_summary.to_excel(writer, index=False, sheet_name='汇总')
        df_detail.to_excel(writer, index=False, sheet_name='详细事件')

    print(f'📄 报告已生成: {file_path}')


if __name__ == '__main__':
    generate_report()
