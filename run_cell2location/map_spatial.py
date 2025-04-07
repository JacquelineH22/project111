import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import cell2location
    from cell2location.utils.filtering import filter_genes
    from cell2location.models import RegressionModel

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def map_spatial(adata_st, inf_aver, save_dir):
    '''
    map spatial data to reference signatures

    Args:
        adata_st: AnnData, spatial data
        inf_aver: pd.DataFrame, reference signatures
        save_dir: str, save directory
    
    Returns:
        adata_st: AnnData, spatial data with cell abundance
        mod: cell2location.models.Cell2location, model

    '''    
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # find shared genes and subset both anndata and reference signatures
    intersect = np.intersect1d(adata_st.var_names, inf_aver.index)
    adata_st = adata_st[:, intersect].copy()
    inf_aver = inf_aver.loc[intersect, :].copy()

    # create the model
    cell2location.models.Cell2location.setup_anndata(adata=adata_st, batch_key=None)
    mod = cell2location.models.Cell2location(
        adata_st, cell_state_df=inf_aver,
        # the expected average cell abundance: tissue-dependent
        # hyper-prior which can be estimated from paired histology:
        N_cells_per_location=5,
        # hyperparameter controlling normalisation of
        # within-experiment variation in RNA detection:
        detection_alpha=20
    )
    mod.view_anndata_setup()

    # train
    mod.train(
        max_epochs=30000,
        # train using full data (batch_size=None)
        batch_size=None,
        # use all data points in training because
        # we need to estimate cell abundance at all locations
        train_size=1
    )
    mod.plot_history(1000)
    plt.show()

    # export the estimated cell abundance (summary of the posterior distribution)
    adata_st = mod.export_posterior(adata_st, sample_kwargs={'num_samples': 1000, 'batch_size': mod.adata.n_obs})

    # add 5% quantile, representing confident cell abundance, 'at least this amount is present', to adata.obs
    adata_st.obs[adata_st.uns['mod']['factor_names']] = adata_st.obsm['q05_cell_abundance_w_sf']
    
    # save
    adata_st.obs[adata_st.uns['mod']['factor_names']].to_csv('%s/deconv.csv' % save_dir)
    mod.save(save_dir, overwrite=True)
    adata_st.write('%s/st.h5ad' % save_dir)

    mod.plot_QC()
    plt.show()

    return adata_st, mod


if __name__ == '__main__':

    st_h5_file = 'path_to_10x_h5_file'
    ref_dir = 'path_to_reference_signatures'
    save_dir = 'path_to_save_dir'

    # load data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata_st = sc.read_10x_h5(st_h5_file)
        adata_st.var_names_make_unique()

    adata_ref = sc.read_h5ad('%s/sc.h5ad' % ref_dir)
    mod = cell2location.models.RegressionModel.load(ref_dir, adata_ref)

    # export estimated expression in each cluster
    if 'means_per_cluster_mu_fg' in adata_ref.varm.keys():
        inf_aver = adata_ref.varm['means_per_cluster_mu_fg'][
            ['means_per_cluster_mu_fg_%s' % i
            for i in adata_ref.uns['mod']['factor_names']]].copy()
    else:
        inf_aver = adata_ref.var[['means_per_cluster_mu_fg_%s' % i
                                for i in adata_ref.uns['mod']['factor_names']]].copy()
    inf_aver.columns = adata_ref.uns['mod']['factor_names']

    # find mitochondria-encoded (MT) genes
    adata_st.var['MT_gene'] = [gene.startswith('MT-') for gene in adata_st.var_names]
    adata_st.obsm['MT'] = adata_st[:, adata_st.var['MT_gene'].values].X.toarray()
    adata_st = adata_st[:, ~adata_st.var['MT_gene'].values]

    map_spatial(adata_st, inf_aver, save_dir)

    print('Done!')
