import pyarrow.parquet as pq
import pyarrow.compute as pc
import time

path = "processed/dataset.parquet"

# 1 — load full dataset
print("=== Full load ===")
start = time.perf_counter()
table = pq.read_table(path)
elapsed = (time.perf_counter() - start) * 1000
print(f"Rows: {table.num_rows}, Columns: {table.num_columns}")
print(f"Load time: {elapsed:.2f} ms")

# 2 — column pruning: only load text + label, skip id
print("\n=== Column pruning (text + label only) ===")
start = time.perf_counter()
table_pruned = pq.read_table(path, columns=["text", "label"])
elapsed = (time.perf_counter() - start) * 1000
print(f"Columns: {table_pruned.column_names}")
print(f"Load time: {elapsed:.2f} ms")

# 3 — predicate pushdown: only load label=1 rows
print("\n=== Predicate pushdown (label == 1) ===")
start = time.perf_counter()
table_filtered = pq.read_table(path, filters=[("label", "=", 1)])
elapsed = (time.perf_counter() - start) * 1000
print(f"Rows after filter: {table_filtered.num_rows}")
print(f"Load time: {elapsed:.2f} ms")

# 4 — convert to format ready for training
print("\n=== Training-ready format ===")
texts = table_filtered.column("text").to_pylist()
labels = table_filtered.column("label").to_pylist()
for text, label in zip(texts, labels):
    print(f"[{label}] {text[:60]}...")