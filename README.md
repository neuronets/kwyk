# kwyk
Knowing what you know - Bayesian Neural Network for brain parcellation and uncertainty estimation

Paper, code, and model corresponding to [preprint](https://arxiv.org/abs/1812.01719)

To run using singularity, first pull the image:

```
singularity pull --docker-login docker://neuronets/kwyk:latest-gpu
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

This will generate two sets of files `output_*.nii.gz` and `output_*_orig.nii.gz`. The first set consists of results in conformed FreeSurfer space. The second set will correspond to the original input space.

1. `output_means`: This file contains the labels
2. `output_variance`: This file contains the variance in labeling over multiple samplings.
3. `output_entropy`: This file contains the estimated entropy at each voxel.

For now, if output files exist, the program will not override them.

### Docker usage example

Instead of singularity with GPU, once can also use docker directly. This is an example with a CPU. Note that the CPU-based run is significantly slower.

```
docker run -it --rm -v $(pwd):/data neuronets/kwyk:latest-cpu -m bvwn_multi_prior --save-entropy T1_001.nii.gz output.nii.gz
```

The above examples assume there is a file named `T1_001.nii.gz` in `$(pwd)`.

# nobrainer

This model is based on an earlier version of the nobrainer framework. This repository will be updated when the code is transitioned to the new model.
