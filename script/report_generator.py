# 文件名: report_generator.py
# 描述: 封装了日报生成AI逻辑的独立模块。

import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 设置日志
logger = logging.getLogger(__name__)

class ReportGeneratorService:
    """
    一个单例服务，用于加载和管理日报生成的AI模型。
    """
    _instance = None
    _model = None
    _tokenizer = None

    def __new__(cls):
        if cls._instance is None:
            logger.info("Creating ReportGeneratorService instance...")
            cls._instance = super(ReportGeneratorService, cls).__new__(cls)
            # 在首次创建实例时，加载模型和分词器
            try:
                logger.info("Loading Hugging Face model 'google/flan-t5-base' for report generation...")
                cls._tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
                cls._model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
                logger.info("✅ Report generation model loaded successfully.")
            except Exception as e:
                logger.error(f"❌ Failed to load report generation model: {e}")
                cls._model = None
                cls._tokenizer = None
        return cls._instance

    def build_prompt(self, summary: dict) -> str:
        """根据数据摘要构建给AI模型的提示。"""
        prompt = """你是一个安防监控系统的智能助手，请根据以下数据生成一份简明的中文监控日报。
包括：总体事件概况、各类型事件情况、摄像头在线状态，以及必要时的风险提示。

下面是当天的监控数据摘要：
"""
        key_map = {
            '日期': '日期',
            '总事件数': '总事件数',
            '未处理事件数': '未处理事件数',
            '处理中事件数': '处理中事件数',
            '已处理事件数': '已处理事件数',
            '摄像头总数': '摄像头总数',
            '在线摄像头': '在线摄像头',
        }
        for key, label in key_map.items():
            if key in summary:
                prompt += f"- {label}: {summary[key]}\n"

        if '各类型事件统计' in summary and summary['各类型事件统计']:
            prompt += "- 各类型事件统计:\n"
            for event_type, count in summary['各类型事件统计'].items():
                prompt += f"  - {event_type}: {count} 次\n"

        prompt += "\n请输出一段自然语言描述，总结当天监控情况。\n"
        return prompt.strip()

    def generate_text(self, prompt: str) -> str:
        """使用已加载的AI模型生成报告文本。"""
        if not self._model or not self._tokenizer:
            raise RuntimeError("Report generation AI model is not available.")

        inputs = self._tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self._model.generate(**inputs, max_new_tokens=256)
        content = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return content


# 在模块加载时，创建服务单例，这将自动加载模型
try:
    report_service_instance = ReportGeneratorService()
except Exception as e:
    # 这里的异常会在Django启动时就被捕获和记录
    report_service_instance = None


def process_report_generation(summary_data: dict) -> str:
    """
    提供给外部调用的标准接口函数。
    接收汇总数据，返回生成的报告文本。
    """
    if report_service_instance is None:
        raise ConnectionError("ReportGeneratorService failed to initialize.")

    # 1. 构建Prompt
    prompt = report_service_instance.build_prompt(summary_data)

    # 2. 生成文本
    report_content = report_service_instance.generate_text(prompt)

    return report_content