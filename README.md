# kwyk
Knowing what you know - Bayesian Neural Network for brain parcellation and uncertainty estimation

Paper, code, and model corresponding to [preprint](https://arxiv.org/abs/1812.01719)

To run using singularity, first pull the image:

```
singularity pull --docker-login docker://satra/kwyk:latest-gpu
```

You have a few options when running the image. To see them call help.
```
singularity run -B $(pwd):/data -W /data --nv kwyk_latest-gpu.sif --help
```

The models correspond to:
1. Spike-and-slab dropout (bvwn_multi_prior)
2. MC Bernoulli dropout (bwn_multi)
3. MAP (bwn)

Here is an example with the spike and slab dropout.
```
singularity run -B $(pwd):/data -W /data --nv kwyk_latest-gpu.sif -m bvwn_multi_prior -n 2 --save-variance --save-entropy T1_001.nii.gz output.nii.gz
```

For now, if output files exist, the program will not override them.

# nobrainer

This model is based on an earlier version of the nobrainer framework. This repository will be updated when the code is transitioned to the new model.
