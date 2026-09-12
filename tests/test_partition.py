import pandas as pd

rows = []
for i, df in enumerate(all_client_dfs):  # đúng 10 phần tử
    n_total = len(df)
    n_pos = df["TARGET"].sum()
    rows.append({
        "client": i,
        "n_total": n_total,
        "n_positive": n_pos,
        "target_rate": n_pos / n_total,
        "share_of_global_positive": n_pos / 24825,
    })

summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
print("\nTổng positive đã gán:", summary["n_positive"].sum(), "(phải = 24825)")
print("Client có 0 positive:", (summary["n_positive"] == 0).sum())