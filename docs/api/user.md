# User API

This page mirrors the public, user-facing package surface. Each entry links to an auto-generated detail page built from the current source code and docstrings.

## Core Models

```{eval-rst}
.. currentmodule:: navigo

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: class_no_inherited.rst

   Navigo
   MLPTimeGRN
```

## High-Level Workflows

```{eval-rst}
.. currentmodule:: navigo

.. autosummary::
   :toctree: generated
   :nosignatures:

   run_perturbation_inference
   set_seed
```

## Input And Preprocessing

```{eval-rst}
.. currentmodule:: navigo.io

.. autosummary::
   :toctree: generated
   :nosignatures:

   read_h5ad
   load_and_preprocess_data
   normalize_to_x
   to_dense
   gene_names_from_var
```

```{eval-rst}
.. currentmodule:: navigo.pp

.. autosummary::
   :toctree: generated
   :nosignatures:

   load_atlas
```

## Trajectory And Analysis Tools

```{eval-rst}
.. currentmodule:: navigo.tl

.. autosummary::
   :toctree: generated
   :nosignatures:

   interpolate_atlas
   evaluate_interpolation
   build_interpolation_umap
   denoise_trajectory
   compute_deg_by_day
   extract_denoising_markers
   denoising_pathway_enrichment
   build_denoising_marker_table
   build_denoising_pathway_table
   build_denoising_trajectory_table
   compute_grn_expression_changes
   run_grn_analysis_scripts
   evaluate_reprogramming_screen
   parse_training_log
   summarize_round_scores
   sample_training_subset
   validate_training
```

```{eval-rst}
.. currentmodule:: navigo.trajectory

.. autosummary::
   :toctree: generated
   :nosignatures:

   prepare_time_axis
   interior_test_times
   neighboring_train_times
   transfer_labels
   subset_for_time
   sample_index_array
   collapse_msmu
   get_norm_msmu
   extract_gene_expression
```

## Plotting And Network Visualization

```{eval-rst}
.. currentmodule:: navigo.pl

.. autosummary::
   :toctree: generated
   :nosignatures:

   interpolation_umap
   stacked_bar
   expression_heatmap
   enrichment_barh
   grouped_barh
   marker_ratio_boxplot
   perturbation_effect_plot
```

```{eval-rst}
.. currentmodule:: navigo.network

.. autosummary::
   :toctree: generated
   :nosignatures:

   collect_edges
   plot_three_layer_network
```

## GRN-Specific Utilities

```{eval-rst}
.. currentmodule:: navigo.grn

.. autosummary::
   :toctree: generated
   :nosignatures:

   load_ko_responses
   cluster_and_embed
   top_genes_distance_matrix
   chd_cluster_distribution
   pathway_enrichment_comparison
   marker_change_analysis
   marker_change_from_jacobian
   interaction_network_data
```

## Evaluation Metrics

```{eval-rst}
.. currentmodule:: navigo.metrics

.. autosummary::
   :toctree: generated
   :nosignatures:

   bh_fdr
   wilcoxon_deg
   signature_overlap
   distribution_metrics
   cell_type_metrics
```

## Utility Functions

```{eval-rst}
.. currentmodule:: navigo.utils

.. autosummary::
   :toctree: generated
   :nosignatures:

   vis_log
   calculate_distance
   generate_alignment_cell
   matching_forward
   set_seed
```
