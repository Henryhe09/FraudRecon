"""Narrative input fields and internal sample identifiers."""
import json
from pathlib import Path


def load_narratives(input_path: Path):
    """Read narratives; generate internal IDs when the caller supplies none."""
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"输入文件应为 narrative 对象的 JSON 数组: {input_path}")
    reserved = set()
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index + 1} 条输入应为对象")
        values = [str(item[k]).strip() for k in ('case_id','案件编号') if item.get(k) is not None and str(item[k]).strip()]
        if len(set(values)) > 1:
            raise ValueError(f"第 {index + 1} 条 case_id 与案件编号不一致")
        if values:
            cid = values[0]
            if cid in reserved: raise ValueError(f"重复的样本编号: {cid}")
            if Path(cid).name != cid or cid in ('.','..'):
                raise ValueError(f"样本编号不能包含目录路径: {cid}")
            reserved.add(cid)
    cases = []; next_id = 1
    for index, item in enumerate(data):
        text = item.get('narrative', item.get('案情描述'))
        if not isinstance(text,str) or not text.strip():
            raise ValueError(f"第 {index + 1} 条缺少非空 narrative")
        cid = next((str(item[k]).strip() for k in ('case_id','案件编号') if item.get(k) is not None and str(item[k]).strip()), None)
        if cid is None:
            while f'case_{next_id:06d}' in reserved: next_id += 1
            cid = f'case_{next_id:06d}'; reserved.add(cid); next_id += 1
        record = dict(item)
        record['案件编号'] = cid
        record['案情描述'] = text
        record['案件类别'] = item.get('category') or item.get('案件类别')
        cases.append(record)
    return cases

