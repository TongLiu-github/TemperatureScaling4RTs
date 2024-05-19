# Temperature-scaling surprisal estimates improve fit to human reading times – but does it do so for the “right reasons”?

Code for the paper [Temperature-scaling surprisal estimates improve fit to human reading times – but does it do so for the “right reasons”?](https://openreview.net/pdf?id=zdAXhcemNF) (ACL 2024 long paper).



[<img src="https://github.com/TongLiu-github/temperature-scaling-for-fit-to-reading-times/blob/main/figures/optimal_T_gpt2_naturalstories.png" alt="viewer" width="400">](https://prismarinejs.github.io/prismarine-viewer/)
[<img src="https://github.com/TongLiu-github/temperature-scaling-for-fit-to-reading-times/blob/main/figures/optimal_T_gpt2_brown.png" alt="viewer" width="400">](https://prismarinejs.github.io/prismarine-viewer/)

<h3> Prepare Data </h3>
Processed data for Natural Stories and Brown:  

./PPP_Calculation_Natural_Stories/data/all.txt.annotation.filtered.csv,  
and  
./PPP_Calculation_Brown/data/all.txt.annotation.filtered.csv.

<h3> How to run </h3>

GPT2 s/m/l/xl on Natural Stories/Brown corpora:   

```bash
sh run.sh
```

Note: Core Temperature-scaling code is at line 230-231 in PPP_calculation.py:  
```bash
pred_test_logits1 = pred_test_logits0 / T
cal_test_probs1 = pred_test_logits1.softmax(-1)
```

