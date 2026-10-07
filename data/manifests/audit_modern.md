# Audit dataset modern

- total gambar: 16,787
- dibuang saringan CLIP (bukan foto wajah manusia, p < 0.5): 321
- duplikat md5 dibuang: 3 (grup duplikat lintas split: 2)
- pemetaan label: {'real': 0, 'fake': 1}

## Jumlah per split dan grup

```
label_name            fake  real
split group                     
test  dff              793   797
      modern_baru      821     0
      modern_dikenal     0    18
      stylegan         500   499
train dff             2965  2978
      stylegan        1998  1995
valid dff              294   292
      modern          1139  1298
      stylegan         200   200
```

## Generator di train vs test

- generator fake di train: ['sd-1.5 (DFF)', 'stylegan']
- generator di grup modern_baru (tidak ada di train): ['flux.2-klein-9b', 'gpt-image-1.5', 'gpt-image-2', 'ideogram-2.0', 'midjourney-7', 'nano-banana-pro', 'recraft-v3', 'seedream-v5.0', 'z-image-turbo']

Rincian per sumber: `modern_per_source.csv`.
