import json
import pyarrow as pa
import pyarrow.parquet as pq
import time

# read raw jsonl
print("Reading raw data...")
records = []
with open("raw/data.jsonl", "r") as f:
    for line in f:
        records.append(json.loads(line.strip()))

print(f"Loaded {len(records)} records")

# convert to arrow table
ids     = pa.array([r["id"]    for r in records], type=pa.int32())
texts   = pa.array([r["text"]  for r in records], type=pa.string())
labels  = pa.array([r["label"] for r in records], type=pa.int32())

schema = pa.schema([
    pa.field("id",    pa.int32()),
    pa.field("text",  pa.string()),
    pa.field("label", pa.int32()),
])

table = pa.table({"id": ids, "text": texts, "label": labels}, schema=schema)

print(f"Arrow table shape: {table.num_rows} rows x {table.num_columns} columns")
print(f"Schema:\n{table.schema}")

# write to parquet
output_path = "processed/dataset.parquet"
pq.write_table(table, output_path, compression="snappy")

import os
size_kb = os.path.getsize(output_path) / 1e3
print(f"\nWritten to {output_path} ({size_kb:.1f} KB)")
print("Compression: snappy")

# read back and verify
print("\nVerifying...")
table_back = pq.read_table(output_path)
print(f"Read back {table_back.num_rows} rows")
print(f"First row: {table_back.slice(0, 1).to_pydict()}")