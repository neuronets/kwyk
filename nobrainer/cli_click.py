from pathlib import Path
import subprocess
import tempfile

import click
import nibabel as nib
import numpy as np

from nobrainer.io import read_volume
from nobrainer.predict import _get_predictor
from nobrainer.predict import predict_from_filepath
from nobrainer.volume import from_blocks
from nobrainer.volume import to_blocks
from nobrainer.volume import zscore

_here = Path(__file__).parent

_models = {
    'bwn': _here.parent / 'saved_models' / 'all_50_wn' / '1555341859',
    'bwn_multi': _here.parent / 'saved_models' / 'all_50_bwn_09_multi' / '1555963478',
    'bvwn_multi_prior': _here.parent / 'saved_models' / 'all_50_bvwn_multi_prior' / '1556816070',
}

@click.command()
@click.argument('infile')
@click.argument('outfile')
@click.option('-m', '--model', type=click.Choice(_models.keys()), default="bwn_multi", required=True, help='Model to use for prediction.')
@click.option('-n', '--n-samples', type=int, default=1, help='Number of samples to predict.')
@click.option('-b', '--batch-size', type=int, default=8, help='Batch size during prediction.')
@click.option('--save-variance', is_flag=True, help='Save volume with variance across `n-samples` predictions.')
@click.option('--save-entropy', is_flag=True, help='Save volume of entropy values.')
def predict(*, infile, outfile, model, n_samples, batch_size, save_variance, save_entropy):
    """Predict labels from features using a trained model.

    The predictions are saved to OUTFILE.

    If you encounter out-of-memory issues, use a lower batch size value.
    """
    _orig_infile = infile
    _orig_outfile = outfile

    # Are there other neuroimaging file extensions with multiple periods?
    if outfile.lower().endswith('.nii.gz'):
        outfile_ext = '.nii.gz'
        outfile_stem = outfile[:-7]
    else:
        outfile_ext = Path(outfile).suffix
        outfile_stem = Path(outfile).stem

    outfile_means = "{}_means{}".format(outfile_stem, outfile_ext)
    outfile_variance = "{}_variance{}".format(outfile_stem, outfile_ext)
    outfile_entropy = "{}_entropy{}".format(outfile_stem, outfile_ext)

    for ff in [outfile_means, outfile_variance, outfile_entropy]:
        if Path(ff).exists():
            raise FileExistsError("file exists: {}".format(ff))

    required_shape = (256, 256, 256)
    block_shape = (32, 32, 32)

    img = nib.load(infile)
    ndim = len(img.shape)
    if ndim != 3:
        raise ValueError("Input volume must have three dimensions but got {}.".format(ndim))
    if img.shape != required_shape:
        tmp = tempfile.NamedTemporaryFile(suffix='.nii.gz')
        print("++ Conforming volume to 1mm^3 voxels and size 256x256x256.")
        _conform(infile, tmp.name)
        infile = tmp.name
    else:
        tmp = None

    savedmodel_path = _models[model]

    print("++ Running forward pass of model.")
    predictor = _get_predictor(savedmodel_path)
    outputs = predict_from_filepath(
        infile,
        predictor=predictor,
        block_shape=block_shape,
        return_variance=True,
        return_entropy=True,
        n_samples=n_samples,
        batch_size=batch_size,
        normalizer=zscore)

    # Delete temporary file.
    if tmp is not None:
        tmp.close()

    if n_samples > 1:
        means, variance, entropy = outputs
    else:
        means, entropy = outputs
        variance = None

    outfile_means_orig = "{}_means_orig{}".format(outfile_stem, outfile_ext)
    outfile_variance_orig = "{}_variance_orig{}".format(outfile_stem, outfile_ext)
    outfile_entropy_orig = "{}_entropy_orig{}".format(outfile_stem, outfile_ext)

    print("++ Saving results.")
    data = np.round(means.get_fdata()).astype(np.uint8)
    means = nib.Nifti1Image(data, header=means.header, affine=means.affine)
    means.header.set_data_dtype(np.uint8)
    nib.save(means, outfile_means)
    _reslice(outfile_means, outfile_means_orig, _orig_infile, True)
    if save_variance and variance is not None:
        nib.save(variance, outfile_variance)
        _reslice(outfile_variance, outfile_variance_orig, _orig_infile)
    if save_entropy:
        nib.save(entropy, outfile_entropy)
        _reslice(outfile_entropy, outfile_entropy_orig, _orig_infile)


def _conform(input, output):
    """Conform volume using FreeSurfer."""
    subprocess.run(['mri_convert', '--conform', input, output], check=True)
    return output

def _reslice(input, output, reference, labels=False):
    """Conform volume using FreeSurfer."""
    if labels:
        subprocess.run(['mri_convert', '-rl', reference, '-rt', 'nearest', '-ns', '1', 
                        input, output], 
                       check=True)
    else:
        subprocess.run(['mri_convert', '-rl', reference, input, output], check=True)
    return output
