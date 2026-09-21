# examples/

示例输入。`example.fasta` 为 3 条病毒序列子集，用于快速跑通流程。

```bash
python predict.py \
  --config configs/vhe_net_with_weight.yaml \
  --checkpoint checkpoints/vhe_net_with_weight/last_model.pt \
  --out outputs/example_predictions.csv
```
