# for n in "1000" 
# data_name = "Brown", "Natural_Stories"
# model_name0 = "gpt2" "gpt2_medium" "gpt2_large" "gpt2_xl" 
for data_name0 in "Brown" 
do
    for model_name0 in "gpt2" 
    do
        python data_processing_for_surprisal.py -n 1000 -data_name ${data_name0}
        python logits_calculation.py -n 1000 -data_name ${data_name0} -model_name ${model_name0} -cuda_num "0"
        python PPP_calculation.py -n 1000 -data_name ${data_name0} -model_name ${model_name0} -cuda_num "0"  -K 10 -T_optimal 1.0
        # python PPP_calculation.py -n 1000 -data_name ${data_name0} -model_name ${model_name0} -cuda_num "0"  -K 0 
    done
done
