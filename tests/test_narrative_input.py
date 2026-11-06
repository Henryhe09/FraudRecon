import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from ndrr.fpe import extract as fpe

class NarrativeInputTests(unittest.TestCase):
    def load(self,rows):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'narratives.json';p.write_text(json.dumps(rows,ensure_ascii=False))
            return fpe.load_cases(p)
    def test_narrative_only_gets_stable_internal_ids(self):
        rows=[{'narrative':'第一段叙述'},{'narrative':'第二段叙述'}]
        a=self.load(rows);b=self.load(rows)
        self.assertEqual(a,b)
        self.assertEqual([x['案件编号'] for x in a],['case_000001','case_000002'])
        self.assertEqual(a[0]['案情描述'],'第一段叙述')
    def test_supplied_ids_are_preserved_and_reserved(self):
        rows=[{'narrative':'甲'},{'case_id':'case_000001','narrative':'乙'},{'案件编号':'original_2','案情描述':'丙','案件类别':'类别'}]
        cases=self.load(rows)
        self.assertEqual([x['案件编号'] for x in cases],['case_000002','case_000001','original_2'])
        self.assertEqual(cases[-1]['案件类别'],'类别')
    def test_duplicate_ids_and_missing_narrative_are_rejected(self):
        for rows in [[{'case_id':'a','narrative':'甲'},{'case_id':'a','narrative':'乙'}],[{'narrative':' ' }]]:
            with self.assertRaises(ValueError):self.load(rows)
    def test_number_is_metadata_not_model_prompt_and_category_can_be_inferred(self):
        text='同一段案情叙述'
        a=self.load([{'narrative':text}])[0]
        b=self.load([{'案件编号':'provided','案情描述':text}])[0]
        with patch.object(fpe,'call_api',return_value=json.dumps({'fraud_type':'冒充电商物流客服类'})) as call:
            first=fpe.process_one_case(0,a,1);request_a=call.call_args
            second=fpe.process_one_case(0,b,1);request_b=call.call_args
        self.assertEqual(request_a,request_b)
        self.assertEqual(first['案件类别'],'冒充电商物流客服类')
        self.assertEqual(second['案件编号'],'provided')
