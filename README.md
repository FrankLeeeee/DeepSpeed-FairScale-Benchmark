# DeepSpeed-FairScale-Benchmark

## Run Benchmarking

```shell
# run both deepspeed and fairscale
bash ./scripts/run_all.sh

# run deepspeed
bash ./scripts/run_deepspeed.sh

# run fairscale
bash ./scripts/run_fairscale.sh

```

## Results

On DGX with 8 A100 (80 GB), the benchmarking results are as follows:

<table>
    <tr>
        <td rowspan="2">Stage</td>
        <td colspan="3">Step time / sec</td>
        <td colspan="3">GPU RAM Max Allocated / GB</td>
    </tr>
    <tr>
        <td>DeepSpeed</td>
        <td>FairScale</td>
        <td>PyTorch Native</td>
        <td>DeepSpeed</td>
        <td>FairScale</td>
        <td>PyTorch Native</td>
    </tr>
    <tr>
        <td>1</td>
        <td>0.799</td>
        <td>0.812</td>
        <td>0.727</td>
        <td>27.0</td>
        <td>27.8</td>
        <td>27.8</td>
    </tr>
    <tr>
        <td>2</td>
        <td>0.798</td>
        <td>0.817</td>
        <td>-</td>
        <td>27.9</td>
        <td>26.9</td>
        <td>-</td>
    </tr>
    <tr>
        <td>3</td>
        <td>0.984</td>
        <td>0.809</td>
        <td>-</td>
        <td>28.9</td>
        <td>28.0</td>
        <td>-</td>
    </tr>
</table>