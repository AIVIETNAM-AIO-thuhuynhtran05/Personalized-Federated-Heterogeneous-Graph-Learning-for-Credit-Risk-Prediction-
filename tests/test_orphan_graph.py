import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from src.graph.orphan_handler import prepare_transaction_nodes
from src.graph.node_builder import build_nodes
from src.graph.edge_builder import build_edges
from src.graph.heterograph_builder import build_client_graph


class OrphanGraphTests(unittest.TestCase):
    def tables(self):
        tables = {
            "application_train": pd.DataFrame({"SK_ID_CURR": [1, 2], "TARGET": [0, 1]}),
            "bureau": pd.DataFrame({"SK_ID_CURR": [1], "SK_ID_BUREAU": [10]}),
            "previous_application": pd.DataFrame({"SK_ID_CURR": [1, 2], "SK_ID_PREV": [100, 200]}),
        }
        for table in ("installments_payments", "POS_CASH_balance", "credit_card_balance"):
            tables[table] = pd.DataFrame({"SK_ID_CURR": [1, 2, 1, 99, None, 1],
                                         "SK_ID_PREV": [100, 999, None, 999, 100, 200]})
        return tables

    def test_fallback_exclusive_with_parent_edge_and_quarantine(self):
        tables, quarantine = prepare_transaction_nodes(self.tables())
        nodes = build_nodes(tables)
        edges, missing = build_edges(nodes)
        for node, table in (("installment", "installments_payments"), ("pos_cash", "POS_CASH_balance"), ("credit_card", "credit_card_balance")):
            self.assertEqual(tables[table].is_orphan_prev.tolist(), [0, 1, 1])
            self.assertEqual(nodes[node].source_row_id.tolist(), [0, 1, 2])
            self.assertEqual(quarantine[table].source_row_id.tolist(), [3, 4, 5])
            self.assertEqual(quarantine[table].quarantine_reason.tolist(),
                             ["customer_not_in_client", "customer_not_in_client", "previous_customer_mismatch"])
            normal = edges[f"previous_application__has__{node}"]
            fallback = edges[f"customer__has_orphan__{node}"]
            self.assertEqual(normal[1].tolist(), [0])
            self.assertEqual(fallback[1].tolist(), [1, 2])
            self.assertEqual(missing[node], 2)
            np.testing.assert_array_equal(fallback[::-1], edges[f"{node}__rev_has_orphan__customer"])
            self.assertNotIn(f"customer__has__{node}", edges)

    def test_quarantine_is_persisted_and_source_rows_remain_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = root / "client_000"
            client.mkdir()
            tables = self.tables()
            for table, frame in tables.items():
                frame.to_csv(client / f"{table}.csv", index=False)
            report = build_client_graph(client, root / "graphs/client_000", {k: v.columns.tolist() for k,v in tables.items()})
            self.assertEqual(report["node_counts"]["installment"], 3)
            self.assertEqual(report["quarantine"]["installments_payments"]["rows"], 3)
            self.assertEqual(len(pd.read_csv(client / "installments_payments.csv")), 6)
            self.assertEqual(len(pd.read_csv(root / "graphs/quarantine/client_000/installments_payments.csv")), 3)


if __name__ == "__main__":
    unittest.main()
