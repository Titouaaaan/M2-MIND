import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# change this path if needed
DATA_DIR = r"C:\Users\titouan\OneDrive\Bureau\M2-MIND\MEDS\PART 1\WikiTableQuestions-1.0.2-compact\WikiTableQuestions"
CSV_DIR = os.path.join(DATA_DIR, "csv")

table_shapes = []
table_info = []  # store info: (table_name, rows, cols, column_names)

for subfolder in os.listdir(CSV_DIR):
    subpath = os.path.join(CSV_DIR, subfolder)
    if not os.path.isdir(subpath):
        continue

    for fname in os.listdir(subpath):
        if not fname.endswith(".tsv"):
            continue
        fpath = os.path.join(subpath, fname)
        try:
            t = pd.read_csv(fpath, sep=None, engine='python', on_bad_lines='skip', encoding='utf-8')
            nrows, ncols = t.shape
            table_shapes.append((nrows, ncols))
            table_info.append({
                'table_name': fpath,
                'rows': nrows,
                'cols': ncols,
                'columns': list(t.columns)
            })
        except Exception as e:
            print(f"Error reading {fpath}: {e}")

print(f"Parsed {len(table_shapes)} tables")

if table_shapes:
    rows = [r for r, _ in table_shapes]
    cols = [c for _, c in table_shapes]

    print(f"Mean rows: {np.mean(rows):.2f}, Mean cols: {np.mean(cols):.2f}")
    print(f"Median rows: {np.median(rows):.2f}, Median cols: {np.median(cols):.2f}")

    # max and min stats 
    max_row_table = max(table_info, key=lambda x: x['rows'])
    min_row_table = min(table_info, key=lambda x: x['rows'])
    max_col_table = max(table_info, key=lambda x: x['cols'])
    min_col_table = min(table_info, key=lambda x: x['cols'])

    print(f"Max rows: {max_row_table['rows']} in table {max_row_table['table_name']}")
    print(f"Min rows: {min_row_table['rows']} in table {min_row_table['table_name']}")
    print(f"Max cols: {max_col_table['cols']} in table {max_col_table['table_name']}")
    print(f"Min cols: {min_col_table['cols']} in table {min_col_table['table_name']}")

    # plot distributions 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # rows
    axes[0].hist(rows, bins=30, color='skyblue', edgecolor='black')
    axes[0].set_title('Distribution of Rows per Table')
    axes[0].set_xlabel('Number of Rows')
    axes[0].set_ylabel('Number of Tables')

    # cols
    axes[1].hist(cols, bins=range(min(cols), max(cols)+2), color='salmon', edgecolor='black', align='left')
    axes[1].set_title('Distribution of Columns per Table')
    axes[1].set_xlabel('Number of Columns')
    axes[1].set_ylabel('Number of Tables')

    plt.tight_layout()    
    plt.savefig("table_size_distributions.png")
    plt.show()

else:
    print("No tables found — check CSV_DIR path.")
