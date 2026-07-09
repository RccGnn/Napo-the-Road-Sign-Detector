import pandas as pd
import shutil
import re
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from Scripts.utilities import utility as u

img = u.get_dataset_dir() / "preprocessed_images"
csv = u.get_dataset_dir() / "preprocessed_images.csv"


df = pd.read_csv(csv)                   # colonne: nome, classe
dataset_root = Path(__file__).parent /  "preprocessed_images"      # struttura attuale: DATASET/<classee>/<nome>
out_dir = Path('Dataset_split/')       # struttura finale: DATASET_SPLIT/<split>/<classee>/<nome>

print(df.head())
print(df.columns)
print(df['classe'].unique()[:10])
print(df['nome'].head())


df['src_path'] = df.apply(lambda r: dataset_root / r['classe'] / r['nome'], axis=1)
df = df[df['src_path'].apply(lambda p: p.exists())].reset_index(drop=True)

df['group'] = df['nome'].apply(lambda f: re.sub(r'_\d+$', '', Path(f).stem))

# Stage 1: separa train (~70%) da temp (~30%)
# n_splits=3 -> ogni fold è ~33% di test, il resto (~67%) è train. Il più vicino a 70/30 con k intero.
sgkf1 = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
train_idx, temp_idx = next(sgkf1.split(df, df['classe'], df['group']))
df_train = df.iloc[train_idx].reset_index(drop=True)
df_temp  = df.iloc[temp_idx].reset_index(drop=True)

# Stage 2: divide temp a metà -> ~15% val, ~15% test
sgkf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
val_idx, test_idx = next(sgkf2.split(df_temp, df_temp['classe'], df_temp['group']))
df_val  = df_temp.iloc[val_idx].reset_index(drop=True)
df_test = df_temp.iloc[test_idx].reset_index(drop=True)

# Sanity check anti-leakage
assert not (set(df_train['group']) & set(df_val['group']))
assert not (set(df_train['group']) & set(df_test['group']))
assert not (set(df_val['group']) & set(df_test['group']))

print(f"Train: {len(df_train)} ({len(df_train)/len(df):.1%})")
print(f"Val:   {len(df_val)} ({len(df_val)/len(df):.1%})")
print(f"Test:  {len(df_test)} ({len(df_test)/len(df):.1%})")

# Sanity check stratificazione: distribuzione classei per split
for name, d in [('train', df_train), ('val', df_val), ('test', df_test)]:
    print(f"\n{name}:")
    print(d['classe'].value_counts(normalize=True).round(3))

for split_name, split_df in [('train', df_train), ('val', df_val), ('test', df_test)]:
    for _, row in split_df.iterrows():
        dest_dir = out_dir / split_name / row['classe']
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(row['src_path'], dest_dir / row['nome'])

print("\nFatto. Struttura pronta in", out_dir)