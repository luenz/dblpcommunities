Subfolders code/no_postprocessing, code/postprocessing, code/postprocessing_avg_degree, code/postprocessing_combined_both and code/postprocessing_with_closer_edge_weights are outdated, only code/all_in_one is up to date.

To use the program, CDlib v2.6.0 is required (as well as some of its dependencies).
Furthermore, dblp.xml and dblp.dtd are required, with dblp.dtd needing to be present in the directory the main functions are called from (parse_dblp_yearly_timeslices.py and show_single_timeslice.py).

community_louvain.py, community_status.py and scan.py were taken from CDlib v2.6.0 and edited slightly to make them more usable for this project. 
community_louvain.py is released under BSD license by THomas Aynaud.
