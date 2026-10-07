# Audit dataset

- image_root: `/Users/calvinn/projects/face-authenticity/data/raw/140k-real-and-fake-faces/real_vs_fake/real-vs-fake`
- gambar valid: 140,000
- gambar rusak/tidak terbaca: 0
- pemetaan internal: {'real': 0, 'fake': 1} (positif = fake)

- Pemetaan label SUMBER (dari CSV): {'fake': [0], 'real': [1]}
- Label CSV vs nama folder tidak cocok: 0
- Split CSV vs folder tidak cocok: 0
- Ada di CSV tapi tidak ada filenya: 0; file tanpa baris CSV: 0

## Jumlah per split

```
label_name   real   fake
split                   
train       50000  50000
valid       10000  10000
test        10000  10000
```

## Ukuran gambar

- 256x256: 140,000

## Duplikasi

```
{'checked': True, 'method': 'md5 isi file (duplikat persis; near-duplicate tidak diperiksa)', 'files_in_duplicate_groups': 0, 'duplicate_groups': 0, 'files_duplicated_across_splits': 0, 'groups_with_conflicting_labels': 0}
```

## Manifest mode `quick`

```
       fake  real
test     16    16
train    32    32
valid    16    16
```

## Manifest mode `subset`

```
        fake   real
test   10000  10000
train   1000   1000
valid    500    500
```
