# Kiểm tra audit phát hiện sai số lần xuất hiện khóa và không nhầm orphan nguồn với lỗi phân bổ.
import importlib.util
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from src.partition.relational_partition import partition_tables

spec = importlib.util.spec_from_file_location("relational_audit", Path(__file__).resolve().parents[1] / "scripts/07_audit_relational_partition.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RelationalAuditTests(unittest.TestCase):
    def test_detects_changed_key_multiplicity_without_confusing_source_orphans(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            pd.DataFrame({"SK_ID_CURR": range(8), "TARGET": [0]*4 + [1]*4}).to_csv(source / "application_train.csv", index=False)
            for name, key in module.RELATED.items():
                data = pd.DataFrame({"SK_ID_CURR": range(8), key: range(100, 108)})
                if name == "installments_payments":
                    data.loc[0, key] = 999
                data.to_csv(source / f"{name}.csv", index=False)
            (source / "missing_report.json").write_text("{}")
            partition = partition_tables(source, root / "clients", number_of_clients=2)
            result = module.audit(source, root / "clients", root / "audit.json", chunksize=2)
            self.assertTrue(result["partition_pass"])
            self.assertEqual(result["tables"]["installments_payments"]["missing_previous_in_source"], 1)
            client = partition["clients"][0]
            path = root / "clients" / client / "installments_payments.csv"
            frame = pd.read_csv(path)
            frame.loc[0, "SK_ID_PREV"] = 99999
            frame.to_csv(path, index=False)
            result = module.audit(source, root / "clients", root / "audit.json", chunksize=2)
            self.assertFalse(result["partition_pass"])
            self.assertEqual(result["tables"]["installments_payments"]["missing_key_occurrences"], 1)
            self.assertEqual(result["tables"]["installments_payments"]["extra_key_occurrences"], 1)


if __name__ == "__main__":
    unittest.main()
