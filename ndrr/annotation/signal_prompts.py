"""Interaction-signal annotation prompt and label definitions."""
import re
from pathlib import Path

RULES = Path(__file__).with_name('signal_rules.txt').read_text()
ALLOWED_SIGNALS = set(re.findall(r'^- ([a-z_]+):', RULES, flags=re.M))
SYSTEM_PROMPT = ('你是一位资深电信反诈案件分析师。请仅根据已重建对话，为每个阶段标注有实际发言支持的策略标签。'
          '不得从原始案情补充对话未出现的信号。标签定义和规则如下：\n' + RULES +
          '\n只输出JSON：{"phases":[{"phase_id":1,"tactics":["标签"]}]}。每个输入阶段恰好一项，未出现信号用空数组。')

