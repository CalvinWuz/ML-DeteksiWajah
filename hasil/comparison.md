# Perbandingan run `full`

Metrik pada split test, threshold 0.5, kelas positif = fake. Checkpoint dipilih dengan val_loss minimum.

| model | tahap | status | accuracy | roc_auc | macro_f1 | f1_real | f1_fake | best_epoch | epochs_run | lr | unfreeze | params_trainable | n_train | device | train_min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vgg16 | A transfer learning | terukur | 0.9389 | 0.9843 | 0.9389 | 0.9381 | 0.9397 | 10 | 10 | 0.0001 | head saja | 6423041 | 100000 | mps | 123.3 |
| vgg16 | B fine-tuning | terukur | 0.9927 | 0.9998 | 0.9927 | 0.9927 | 0.9927 | 10 | 10 | 0.0001 | last_n_modules n=10 | 15862273 | 100000 | mps | 145.2 |
| efficientnet_b0 | A transfer learning | terukur | 0.8377 | 0.9145 | 0.8377 | 0.8365 | 0.8388 | 10 | 10 | 0.0001 | head saja | 1281 | 100000 | mps | 51.6 |
| efficientnet_b0 | B fine-tuning | terukur | 0.9948 | 0.9998 | 0.9947 | 0.9947 | 0.9948 | 8 | 10 | 0.0001 | last_n_modules n=3 | 3157021 | 100000 | mps | 97.5 |
| maxvit_t | A transfer learning | terukur | 0.9404 | 0.9858 | 0.9404 | 0.9399 | 0.9409 | 10 | 10 | 0.0001 | head saja | 264193 | 100000 | mps | 160.9 |
| maxvit_t | B fine-tuning | terukur | 0.9969 | 0.9999 | 0.9968 | 0.9969 | 0.9968 | 10 | 10 | 0.0001 | last_n_modules n=1 | 17794881 | 100000 | mps | 220.6 |
