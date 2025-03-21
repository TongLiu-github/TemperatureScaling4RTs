# Temperature-scaling surprisal estimates improve fit to human reading times – but does it do so for the “right reasons”?

Code for the paper [Temperature-scaling surprisal estimates improve fit to human reading times – but does it do so for the “right reasons”?](https://arxiv.org/pdf/2311.09325) (ACL 2024 long paper).



[<img src="https://github.com/TongLiu-github/TemperatureSaling4RTs/blob/main/figures/optimal_T_gpt2_naturalstories.png" alt="viewer" width="400">](https://prismarinejs.github.io/prismarine-viewer/)
[<img src="https://github.com/TongLiu-github/TemperatureSaling4RTs/blob/main/figures/optimal_T_gpt2_brown.png" alt="viewer" width="400">](https://prismarinejs.github.io/prismarine-viewer/)

<h3> Installation </h3>
To get started, install the package:  

```bash
git clone https://github.com/TongLiu-github/TemperatureSaling4RTs.git
cd TemperatureSaling4RTs
pip install -r requirements.txt
```  

<h3> How to run </h3>

GPT2 s/m/l/xl on Natural Stories/Brown corpora:   

```bash
sh run.sh
```
Results store in *./PPP_Calculation_{corpus}/surprisals/1000/gpt2_{size}/gpt2_{size}__ PPP_result{K}.txt*.  

*Comment 1*: Inside the above script, to calculate the $\Delta_{\mathcal{llh}}$ at $T=1$: 
```bash
python PPP_calculation.py -n 1000 -data_name ${data_name0} -model_name ${model_name0} -cuda_num "0"  -K 10 -T_optimal 1.0
```
At $T\geq1$ (e.g., $T \in [1.0, 10.0]$):  
```bash
python PPP_calculation.py -n 1000 -data_name ${data_name0} -model_name ${model_name0} -cuda_num "0"  -K 0 
```     


*Comment 2*: Core Components of the Temperature-Scaling Code:  
1. Calculate logits, probabilities and labels (utils.py).
2. Scale logits using temperature (line 230-231 in PPP_calculation.py):  

*Comment 3*: For experiments on Dundee, the procedure remains the same as above, while the data size is larger (and therefore not uploaded to this repository). 

<h3> Processed Data </h3>  

We provide processed data for Natural Stories and Brown in *./PPP_Calculation_{corpus}/data/all.txt.annotation.filtered.csv*. 

<h3> BibTeX </h3>  

```bash
@article{liu2023temperature,
  title={Temperature-scaling surprisal estimates improve fit to human reading times--but does it do so for the" right reasons"?},
  author={Liu, Tong and {\v{S}}krjanec, Iza and Demberg, Vera},
  journal={arXiv preprint arXiv:2311.09325},
  year={2023}
}

```   

