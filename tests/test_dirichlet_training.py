# Kiểm thử tích hợp bằng dữ liệu nhỏ: split, chống rò rỉ test, encoder, graph, train và sweep.
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from src.partition.dirichlet_partition import partition_dirichlet, local_split
from src.partition.relational_partition import partition_tables
from src.preprocessing.shared_encoder import fit_shared_encoder, transform_table
from src.graph.heterograph_builder import build_client_graph
from src.graph.graph_dataset import load_graph
from src.models.hetero_gnn import HeteroGNN
from src.federated.client import train_local
from src.federated.server import run_experiment
from src.evaluation.metrics import auc_metrics


class DirichletTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "prepared"
        self.source.mkdir()
        n = 60
        self.customers = pd.DataFrame({"SK_ID_CURR": np.arange(n), "TARGET": [0] * 40 + [1] * 20,
            "amount": np.arange(n, dtype=float), "category": ["A", "B"] * (n // 2)})
        self.customers.to_csv(self.source / "application_train.csv", index=False)
        for table in ("bureau", "previous_application", "installments_payments", "POS_CASH_balance", "credit_card_balance"):
            frame = pd.DataFrame({"SK_ID_CURR": np.arange(n), "value": np.arange(n, dtype=float)})
            frame["SK_ID_BUREAU" if table == "bureau" else "SK_ID_PREV"] = np.arange(n) + 100
            frame.to_csv(self.source / f"{table}.csv", index=False)
        (self.source / "missing_report.json").write_text("{}")

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self):
        clients = self.root / "clients"
        report = partition_tables(self.source, clients, number_of_clients=2, alpha=0.5)
        encoder = fit_shared_encoder(clients, self.root / "encoder.json", chunksize=7)
        graph_dir = self.root / "graphs"
        for client in report["clients"]:
            build_client_graph(clients / client, graph_dir / client, report["schema"], encoder)
        (graph_dir / "manifest.json").write_text(json.dumps({"clients": report["clients"],
            "encoder_fingerprint": encoder["fingerprint"], "partition": {"num_clients": 2, "alpha": 0.5}}))
        return clients, report, encoder, graph_dir

    def test_dirichlet_repeatable_exhaustive_disjoint_and_local_stratification(self):
        clients, a = partition_dirichlet(self.customers, 3)
        _, b = partition_dirichlet(self.customers, 3)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(sorted(a.SK_ID_CURR), list(range(60)))
        self.assertTrue(a.SK_ID_CURR.is_unique)
        for frame in clients.values():
            split = local_split(frame)
            self.assertFalse((split.train_mask & split.test_mask).any())
            for mask in ("train_mask", "test_mask"):
                self.assertEqual(set(frame.iloc[np.flatnonzero(split[mask])].TARGET), {0, 1})

    def test_encoder_ignores_all_test_customer_and_history_values(self):
        clients, report, encoder, _ = self.prepare()
        for client in report["clients"]:
            split = pd.read_csv(clients / client / "customer_split.csv")
            ids = split.loc[split.test_mask, "SK_ID_CURR"]
            for table in report["schema"]:
                path = clients / client / f"{table}.csv"
                frame = pd.read_csv(path)
                mask = frame.SK_ID_CURR.isin(ids)
                if table == "application_train":
                    frame.loc[mask, "amount"] = 1e12
                    frame.loc[mask, "category"] = "TEST_ONLY_CATEGORY"
                else:
                    frame.loc[mask, "value"] = 1e12
                frame.to_csv(path, index=False)
        refit = fit_shared_encoder(clients, self.root / "refit.json", chunksize=7)
        self.assertEqual(encoder, refit)
        spec = encoder["tables"]["application_train"]
        x = transform_table(pd.DataFrame({"amount": [None], "category": ["UNSEEN"]}), spec)
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(x[0, 0], 0.)
        self.assertEqual(x[0, 2], 1.)

    def test_test_labels_and_features_cannot_change_training(self):
        _, report, _, graph_dir = self.prepare()
        graph = load_graph(graph_dir / f"{report['clients'][0]}.npz")
        changed = copy.deepcopy(graph)
        changed["y"][changed["test_mask"]] = 1 - changed["y"][changed["test_mask"]]
        for key in changed:
            if key.startswith("x__"):
                node = key[3:]
                changed[key][changed["test_mask"][changed[f"owner__{node}"]]] = 1e6
        dimensions = {key[3:]: value.shape[1] for key, value in graph.items() if key.startswith("x__")}
        relations = [key[6:] for key in graph if key.startswith("edge__")]
        first = HeteroGNN(dimensions, relations, hidden=4, layers=1)
        second = copy.deepcopy(first)
        train_local(first, graph, batch_size=10)
        train_local(second, changed, batch_size=10)
        for key, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)

    def test_training_logs_both_methods_and_pooled_auc(self):
        _, _, _, graph_dir = self.prepare()
        rows = run_experiment(graph_dir, self.root / "run", rounds=2, batch_size=16, hidden=4, layers=1)
        self.assertEqual(len(rows), 2 * (2 + 1) * 2)
        self.assertEqual({row["method"] for row in rows}, {"federated", "local_only"})
        self.assertTrue(all(row["auc"] is not None for row in rows))
        self.assertTrue((self.root / "run/federated.pt").is_file())
        self.assertIsNone(auc_metrics([0, 0], [0.1, 0.2])["auc"])
        script = Path(__file__).resolve().parents[1] / "scripts/05_evaluate.py"
        spec = importlib.util.spec_from_file_location("evaluation_script", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        final = module.evaluate(self.root / "run")
        for actual, expected in zip(final, rows[-6:]):
            self.assertEqual(actual["auc"], expected["auc"])

    def test_encoded_orphan_flag_and_quarantine_alignment(self):
        clients, report, encoder, graph_dir = self.prepare()
        client = report["clients"][0]
        path = clients / client / "installments_payments.csv"
        frame = pd.read_csv(path)
        frame.loc[frame.index[0], "SK_ID_PREV"] = 999999
        extra = frame.iloc[[0]].copy()
        extra["SK_ID_CURR"] = 999999
        pd.concat([frame, extra], ignore_index=True).to_csv(path, index=False)
        metadata = build_client_graph(clients / client, graph_dir / client, report["schema"], encoder)
        graph = load_graph(graph_dir / f"{client}.npz")
        self.assertEqual(metadata["quarantine"]["installments_payments"]["rows"], 1)
        self.assertEqual(len(graph["x__installment"]), len(frame))
        self.assertEqual(graph["x__installment"].shape[1], encoder["tables"]["installments_payments"]["dimension"] + 1)
        self.assertEqual(graph["is_orphan_prev__installment"].sum(), 1)
        np.testing.assert_array_equal(graph["x__installment"][:, -1], graph["is_orphan_prev__installment"])
        self.assertEqual(graph["edge__customer__has_orphan__installment"][1].tolist(), [0])

    def test_sweep_runs_separate_fits_and_logs_client_count(self):
        script = Path(__file__).resolve().parents[1] / "scripts/06_sweep_clients.py"
        spec = importlib.util.spec_from_file_location("sweep_script", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows = module.sweep(self.source, self.root / "sweep", client_counts=[2, 3],
                            rounds=1, hidden=4, layers=1, batch_size=32)
        self.assertEqual({row["num_clients"] for row in rows}, {2, 3})
        self.assertTrue((self.root / "sweep/sweep_metrics.csv").is_file())

    def test_standalone_local_training_and_checkpoint_predictions(self):
        from src.models.local_training import train_clients
        from src.federated.client import predict_test
        _, report, _, graph_dir = self.prepare()
        output = self.root / "local_training"
        rows = train_clients(graph_dir, output, epochs=2, hidden=4, layers=2,
                             batch_size=16, device="cpu", threads=1)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["status"] == "trained" for row in rows))
        for client in report["clients"]:
            graph = load_graph(graph_dir / f"{client}.npz")
            checkpoint = torch.load(output / client / "model.pt", weights_only=True)
            model = HeteroGNN(**checkpoint["config"])
            model.load_state_dict(checkpoint["model"])
            y, scores = predict_test(model, graph, batch_size=16)
            saved = pd.read_csv(output / client / "test_predictions.csv")
            np.testing.assert_allclose(saved.probability_default, scores)
            np.testing.assert_array_equal(saved.SK_ID_CURR, graph["customer_ids"][graph["test_mask"]])
            history = pd.read_csv(output / client / "history.csv")
            self.assertEqual(history.examples.tolist(), [int(graph["train_mask"].sum())] * 2)
            self.assertTrue(np.isfinite(history.train_loss).all())

    def test_local_training_rejects_overlapping_masks(self):
        from src.graph.validate_training_graph import validate_training_graph
        _, report, _, graph_dir = self.prepare()
        graph = load_graph(graph_dir / f"{report['clients'][0]}.npz")
        manifest = json.loads((graph_dir / "manifest.json").read_text())
        graph["test_mask"][np.flatnonzero(graph["train_mask"])[0]] = True
        with self.assertRaisesRegex(ValueError, "disjoint"):
            validate_training_graph(graph, manifest)


if __name__ == "__main__":
    unittest.main()
