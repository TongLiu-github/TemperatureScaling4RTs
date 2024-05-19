# Temperature-scaling surprisal estimates improve fit to human reading times – but does it do so for the “right reasons”?

Code for the paper [Temperature-scaling surprisal estimates improve fit to human reading times – but does it do so for the “right reasons”?](https://openreview.net/pdf?id=zdAXhcemNF) (ACL 2024 long paper).



[<img src="https://github.com/TongLiu-github/temperature-scaling-for-reading-times/blob/main/figures/optimal_T_gpt2_naturalstories.png" alt="viewer" width="400">](https://prismarinejs.github.io/prismarine-viewer/)
[<img src="https://github.com/TongLiu-github/temperature-scaling-for-reading-times/blob/main/figures/optimal_T_gpt2_brown.png" alt="viewer" width="400">](https://prismarinejs.github.io/prismarine-viewer/)


<h3> How to run </h3>

GPT2 s/m/l/xl on Natural Stories/Brown corpora:   

```bash
sh run.sh
```
 - Hint 1: Inside the above .sh file, to calculate the $\Delta_{\mathcal{llh}}$ at $T=1$: 
```bash
python PPP_calculation.py -n 1000 -data_name ${data_name0} -model_name ${model_name0} -cuda_num "0"  -K 10 -T_optimal 1.0
```
To scale T with $T\geq1$ (e.g., $T \in [1.0, 10.0]$):  
```bash
python PPP_calculation.py -n 1000 -data_name ${data_name0} -model_name ${model_name0} -cuda_num "0"  -K 0 
```     


 - Hint 2: Core Components of the Temperature-Scaling Code:  
1. Calculate logits, probabilities and labels (utils.py).
2. Scale logits using temperature (line 230-231 in PPP_calculation.py):  

<h3> Processed Data </h3>
We provided processed data for Natural Stories and Brown:  

./PPP_Calculation_Natural_Stories/data/all.txt.annotation.filtered.csv,  
and  
./PPP_Calculation_Brown/data/all.txt.annotation.filtered.csv.
