import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import cell2location
    from cell2location.utils.filtering import filter_genes
    from cell2location.models import RegressionModel

import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def read_sc(sc_dir):
    mtx_file = '%s/matrix.mtx' % sc_dir
    barcodes_file = '%s/barcodes.tsv' % sc_dir
    features_file = '%s/features.tsv' % sc_dir
    metadata_file = '%s/metadata.tsv' % sc_dir

    adata = sc.read_mtx(mtx_file).T
    barcodes = pd.read_csv(barcodes_file, sep='\t', header=None)
    features = pd.read_csv(features_file, sep='\t', header=None)
    metadata = pd.read_csv(metadata_file, sep='\t', index_col=0, header=0)

    adata.obs_names = barcodes[0].values
    adata.var_names = features[0].values
    adata.var['gene_symbol'] = features[0].values
    adata.obs = metadata.loc[adata.obs_names]

    adata.var_names_make_unique()

    return adata


def estimate_signatures(adata_sc, save_dir, labels_key='cell_type', batch_key=None, categorical_covariate_keys=None, max_epochs=400):
    selected = filter_genes(adata_sc, cell_count_cutoff=5, cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12)
    adata_sc = adata_sc[:, selected].copy()
    RegressionModel.setup_anndata(
        adata=adata_sc,
        batch_key=batch_key,
        # cell type, covariate used for constructing signatures
        labels_key=labels_key,
        # multiplicative technical effects (platform, 3' vs 5', donor effect)
        categorical_covariate_keys=categorical_covariate_keys
    )
    mod = RegressionModel(adata_sc)
    mod.view_anndata_setup()

    mod.train(max_epochs=max_epochs)
    mod.plot_history(20)
    plt.show()

    adata_sc = mod.export_posterior(
        adata_sc,
        sample_kwargs={'num_samples': 1000, 'batch_size': 2500}
    )

    # save
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    mod.save(save_dir, overwrite=True)
    adata_sc.write('%s/sc.h5ad' % save_dir)

    mod.plot_QC()
    plt.show()

    return adata_sc, mod


if __name__ == '__main__':

    mtx_dir = 'path_to_mtx_dir'
    save_dir = 'path_to_save_dir'

    # load data
    adata_sc = read_sc(mtx_dir)
    adata_sc.X = adata_sc.X.astype('int32')

    # estimate signatures
    adata_sc, mod = estimate_signatures(
        adata_sc,
        save_dir,
        labels_key='cell_type',
        batch_key='sample',
        categorical_covariate_keys=None
    )

    print('Done!')
