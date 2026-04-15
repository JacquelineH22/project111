import tacco as tc
import anndata as ad


def run_tacco(adata_xenium: ad.AnnData, adata_sc: ad.AnnData, cell_type_key: str) -> ad.AnnData:
    """Annotate cell types using TACCO...

    Args:
        adata_xenium (ad.AnnData): AnnData object of Xenium data.
        adata_sc (ad.AnnData): AnnData object of scRNA-seq data.
        cell_type_key (str): Key of cell types in `adata_sc`.

    Returns:
        ad.AnnData: AnnData object of Xenium data with cell type added to `.obs['cell_type_tacco']`.

    """
    tc.preprocessing.construct_reference_profiles(adata_sc, annotation_key=cell_type_key)
    tc.tl.annotate(adata_xenium, adata_sc, cell_type_key, result_key='cell_type_tacco', max_annotation=1)
    adata_xenium.obs['cell_type_tacco'] = adata_xenium.obsm['cell_type_tacco'].idxmax(axis=1)

    return adata_xenium.copy()


if __name__ == '__main__':
    import numpy as np
    from pathlib import Path

    xenium_adata_file = '/data1/hounaiqiao/yy/NucleiSeg/benchmark_review/methods/istar/results/brca_rep1/bin2cell/cells.h5ad'
    sc_adata_file = '/data1/hounaiqiao/wzr/DATA/lhy/brca/sc/tnbc_sc_pp.h5ad'
    save_dir = '/data1/hounaiqiao/yy/NucleiSeg/benchmark_review/methods/istar/results/brca_rep1/annotated/'

    # load data
    print('Loading data...')
    adata_xenium = ad.read_h5ad(xenium_adata_file)
    adata_sc = ad.read_h5ad(sc_adata_file)

    adata_xenium.layers['log1p'] = adata_xenium.X.copy()
    adata_xenium.X = adata_xenium.layers['counts'].copy()
    adata_xenium.X = np.around(adata_xenium.X)

    # tacco
    print('Annotating cell types using TACCO...')
    adata_xenium = run_tacco(adata_xenium, adata_sc, 'label')
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    adata_xenium.write_h5ad('%s/tacco.h5ad' % save_dir)

    print('Done!')
