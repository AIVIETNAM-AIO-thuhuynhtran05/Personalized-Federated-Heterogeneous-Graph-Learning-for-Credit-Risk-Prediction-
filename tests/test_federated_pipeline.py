# Kiểm tra schema, ownership và cạnh ngược xuyên các bước partition → graph trên dữ liệu tổng hợp.
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.partition.relational_partition import partition_tables
from src.graph.heterograph_builder import build_client_graph
from src.graph.node_builder import build_nodes
from src.graph.edge_builder import build_edges


class FederatedPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "prepared"
        self.source.mkdir()
        self.tables = {
            "application_train": pd.DataFrame({"SK_ID_CURR": [1, 2, 3],
                "REGION_RATING_CLIENT_W_CITY": [1, 2, 3],
                "OCCUPATION_TYPE": ["Drivers", "Managers", "Other"], "TARGET": [0, 1, 0]}),
            "bureau": pd.DataFrame({"SK_ID_CURR": [1, 2, 99], "SK_ID_BUREAU": [10, 20, 990]}),
            "previous_application": pd.DataFrame({"SK_ID_CURR": [1, 2], "SK_ID_PREV": [100, 200]}),
        }
        for name in ("installments_payments", "POS_CASH_balance", "credit_card_balance"):
            self.tables[name] = pd.DataFrame({"SK_ID_CURR": [1, 1, 2, 3, 99],
                "SK_ID_PREV": [100, 100, 200, 300, 990], "feature": [None, None, 1., 2., 3.]})
        for name, table in self.tables.items():
            table.to_csv(self.source / f"{name}.csv", index=False)
        (self.source / "missing_report.json").write_text("{}")

    def tearDown(self):
        self.temp.cleanup()

    def test_partition_then_graph_preserves_schema_ownership_and_reverse_edges(self):
        output = self.root / "clients"
        report = partition_tables(self.source, output, number_of_clients=4, chunksize=2, strategy="semantic")
        seen = []
        for client in report["clients"]:
            customers = pd.read_csv(output / client / "application_train.csv")
            seen.extend(customers.SK_ID_CURR.tolist())
            for table, columns in report["schema"].items():
                frame = pd.read_csv(output / client / f"{table}.csv")
                self.assertEqual(frame.columns.tolist(), columns)
                self.assertTrue(set(frame.SK_ID_CURR) <= set(customers.SK_ID_CURR))
            graph_dir = self.root / "graphs" / client
            graph = build_client_graph(output / client, graph_dir, report["schema"])
            self.assertEqual(len(graph["node_counts"]), 6)
            with np.load(graph_dir / "edges.npz") as edges:
                self.assertEqual(len(edges.files), 16)
                for relation in edges.files:
                    src, kind, dst = relation.split("__")
                    if not kind.startswith("rev_"):
                        np.testing.assert_array_equal(edges[relation][::-1], edges[f"{dst}__rev_{kind}__{src}"])
        self.assertEqual(sorted(seen), [1, 2, 3])
        self.assertEqual(report["tables"]["installments_payments"],
                         {"assigned_rows": 4, "unassigned_rows": 1})

    def test_cross_customer_previous_edge_is_rejected(self):
        for name, frame in self.tables.items():
            self.tables[name] = frame.loc[frame.SK_ID_CURR != 99].copy()
        self.tables["installments_payments"].loc[0, "SK_ID_PREV"] = 200
        with self.assertRaisesRegex(ValueError, "different customer"):
            build_edges(build_nodes(self.tables))

    def test_missing_previous_keeps_customer_edge(self):
        for name, frame in self.tables.items():
            self.tables[name] = frame.loc[frame.SK_ID_CURR != 99].copy()
        edges, missing = build_edges(build_nodes(self.tables))
        self.assertEqual(missing["installment"], 1)
        self.assertNotIn("customer__has__installment", edges)
        self.assertEqual(edges["customer__has_orphan__installment"].shape, (2, 1))
        self.assertEqual(edges["previous_application__has__installment"].shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
