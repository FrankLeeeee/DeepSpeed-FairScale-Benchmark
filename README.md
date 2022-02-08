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
        <td>FairScale (Torch AMP) </td>
        <td>PyTorch Native (Torch AMP) </td>
        <td>DeepSpeed</td>
        <td>FairScale (Torch AMP) </td>
        <td>PyTorch Native (Torch AMP) </td>
    </tr>
    <tr>
        <td>1</td>
        <td>0.799</td>
        <td>0.885</td>
        <td>0.800</td>
        <td>27.0</td>
        <td>39.8</td>
        <td>39.8</td>
    </tr>
    <tr>
        <td>2</td>
        <td>0.798</td>
        <td>0.879</td>
        <td>-</td>
        <td>27.9</td>
        <td>37.9</td>
        <td>-</td>
    </tr>
    <tr>
        <td>3</td>
        <td>0.984</td>
        <td>0.869</td>
        <td>-</td>
        <td>28.9</td>
        <td>37.5</td>
        <td>-</td>
    </tr>
</table>