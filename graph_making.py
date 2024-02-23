import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

def angle_grouping(pat_loc):
    pat_loc_phi_sorted = {k: v for k, v in sorted(pat_loc.items(), key=lambda item: item[1][2])}
    pat_loc_theta_sorted = {k: v for k, v in sorted(pat_loc.items(), key=lambda item: item[1][1])}
    groups_phi = {}
    groups_phi_gtv = {}
    reference = list(pat_loc_phi_sorted.values())[1][2] 
    g_num = 'phi_1'
    groups_phi[g_num] = []
    for gtv, l in pat_loc_phi_sorted.items():
        if 0. in l: 
            groups_phi_gtv[gtv] = 'phi_0'
            continue
        phi_diff = l[2] - reference
        if np.abs(phi_diff) < 0.4:
            groups_phi[g_num].append(gtv)
            groups_phi_gtv[gtv] = g_num
        else:
            g_num = f"phi_{int(g_num.split('_')[-1])+1}"
            groups_phi[g_num] = []
            groups_phi[g_num].append(gtv)
            groups_phi_gtv[gtv] = g_num
            reference = l[2]
    for g in groups_phi:
        groups_phi[g] = [k for k in sorted(groups_phi[g], key=lambda gtv: pat_loc[gtv][0])] 
        
    groups_theta = {}
    groups_theta_gtv = {}
    reference = list(pat_loc_theta_sorted.values())[1][1] 
    g_num = 'theta_1'
    groups_theta[g_num] = []
    for gtv, l in pat_loc_theta_sorted.items():
        if 0. in l:
            groups_theta_gtv[gtv] = 'theta_0'
            continue
        theta_diff = l[1] - reference
        if np.abs(theta_diff) < 0.3:
            groups_theta[g_num].append(gtv)
            groups_theta_gtv[gtv] = g_num
        else:
            g_num = f"theta_{int(g_num.split('_')[-1])+1}"
            groups_theta[g_num] = []
            groups_theta[g_num].append(gtv)
            groups_theta_gtv[gtv] = g_num
            reference = l[1]
    for g in groups_theta:
        groups_theta[g] = [k for k in sorted(groups_theta[g], key=lambda gtv: pat_loc[gtv][0])]

    return groups_phi_gtv, groups_theta_gtv


def make_loc_df(pat_loc):

    df_pat = pd.DataFrame(pat_loc)
    df_pat = df_pat.T
    df_pat.columns = ['r', 'theta', 'phi']
    df_pat.sort_values('r', inplace=True)
    
    groups_phi, groups_theta = angle_grouping(pat_loc)
    df_groups_phi = pd.DataFrame(groups_phi.values(), index=groups_phi.keys(), columns=['phi_group'])
    df_groups_theta = pd.DataFrame(groups_theta.values(), index=groups_theta.keys(), columns=['theta_group'])

    df_pat = df_pat.join(df_groups_phi).join(df_groups_theta)
    
    df_pat['x'] = df_pat['r'] * np.sin(df_pat['theta']) * np.cos(df_pat['phi'])
    df_pat['y'] = df_pat['r'] * np.sin(df_pat['theta']) * np.sin(df_pat['phi'])
    df_pat['z'] = df_pat['r'] * np.cos(df_pat['theta'])

    df_pat['r_cyl'] = np.sqrt(df_pat['x']**2 + df_pat['y']**2)
    
    df_source = df_pat[df_pat['phi_group'] == 'phi_0']
    primary = df_source.index[0]
    df_pat.drop(primary, inplace=True)

    return df_pat, df_source


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    return linkage_matrix


def create_node_tree(children, df_pat):
    """
    create list of node connections based on the children from the hierarchical clustering model
    output: list of non-homogeneous nested lists. 
    output[-1] is used as input to the create_connection_tree() function 
    """
    N = len(df_pat)
    def recursive_walk(child):
        node = []
        for ch in child:
            if ch < N:
                node.append(df_pat.index[ch])
            else:
                node.append(recursive_walk(children[ch-N]))

        return node

    all_nodes = []
    for idx in children:
        all_nodes.append(recursive_walk(idx))
    return all_nodes[-1]


def create_connection_tree(tree):
    """
    create tree of connections ordered by branches, based on last entry from create_node_tree() function
    return: dictionary of branches, keys have binary naming convention for successive branching.
    the dict values recursively reference other keys until reaching a terminal branch (leaf)
    dict = { 
             '0': ['00', '01'],
             '1': ['10', '11'],
             ...
             '01': ['010', '011'],
             ...
             '010': 'leaf_name',
           }
    """
    dict_connections = {}
    def recursive_walk(sub_tree, branch):
        print(f"sub: {sub_tree}, branch: {branch}")
        if isinstance(sub_tree, str):
            return
        for i, sub in enumerate(sub_tree):
            sub_branch = branch + str(i)
            if len(sub) == 2:
                dict_connections[sub_branch] = sub
                recursive_walk(sub, sub_branch)
            else:
                dict_connections[sub_branch] = sub

    for i, leaf in enumerate(tree):
        dict_connections[str(i)] = leaf
        recursive_walk(leaf, str(i))

    print(dict_connections)
    terminals = []
    for branch in dict_connections:
        if len(dict_connections[branch]) == 2: continue
        terminals.append(branch)
    terminals_map = {v: k for k, v in dict_connections.items() if k in terminals}
    edges_map = {str(v): k for k, v in dict_connections.items()}

    dict_connections_remap = {}
    for k, v in dict_connections.items():
        dict_connections_remap[k] = []
        if k in terminals:
            dict_connections_remap[k].append(dict_connections[k])
        else:
            for arr in v:
                dict_connections_remap[k].append(edges_map[str(arr)])

    dict_connections_remap = {k: v for k, v in sorted(dict_connections_remap.items(), key=lambda item: len(item[0]))}
    dict_connections_remap = {'p': ['0', '1'], **dict_connections_remap}

    return dict_connections_remap


def make_edges(connections, df_pat, primary):
    edges = []
    skip_to_primary = []
    def process_tree(branch, skip_to_primary):
        def recursive_walk(sub_branch, skip_to_primary):
            compare = []
            sub_branch_leaves = connections[sub_branch]
            for leaf in sub_branch_leaves:
                print(f"        sub leaf: {leaf} for branch: {sub_branch}")
                if leaf in connections.keys():
                    print(f"    entering recursion for {leaf} in {sub_branch}")
                    compare.append(recursive_walk(leaf, skip_to_primary))
                else:
                    compare.append(leaf)
    
            if np.all([isinstance(c, str) for c in compare]):
                df_c = df_pat.loc[compare]
                if len(df_c) == 1:
                    print(f"    exiting recursion for {sub_branch}")
                    return df_c.index[0]
                if abs(df_c.iloc[0]['z'] - df_c.iloc[1]['z']) < 5:
                    df_c.sort_values(['theta_group', 'r'], inplace=True)
                else:
                    df_c.sort_values(['z'], ascending=False, inplace=True)
                print(df_c)
                print(f"1 adding edge {[df_c.index[0], df_c.index[1]]}")
                edges.append([df_c.index[0], df_c.index[1]])
                print(f"    exiting recursion for {sub_branch}")
                return list(df_c.index)
                
            to_return = []
            for i, leaf in enumerate(compare):
                if isinstance(leaf, str):
                    compare[i] = [leaf]
                
            df_0 = df_pat.loc[compare[0]]
            df_0.sort_values('z', ascending=False, inplace=True)
            df_1 = df_pat.loc[compare[1]]
            df_1.sort_values('z', ascending=False, inplace=True)
            print('----------------------')
            print(df_0)
            print(df_1)
            if len(df_0.index) == 2:
                if [df_0.index[0], df_0.index[1]] not in edges:
                    edges.append([df_0.index[0], df_0.index[1]])
            
            if len(df_1.index) == 2:
                if [df_1.index[0], df_1.index[1]] not in edges:
                    print(f"2 adding edge: {[df_1.index[0], df_1.index[1]]}")
                    edges.append([df_1.index[0], df_1.index[1]])
                    
            if df_1['theta_group'].nunique(0) == 1:
                if df_0.loc[df_0.index[-1], 'z'] > df_1.loc[df_1.index[0], 'z'] and np.abs(df_0.loc[df_0.index[-1], 'r_cyl'] - df_1.loc[df_1.index[0], 'r_cyl']) < 20 :
                    if not (df_0.loc[df_0.index[-1], 'phi_group'] != df_1.loc[df_1.index[0], 'phi_group'] and df_0.loc[df_0.index[-1], 'theta_group'] != df_1.loc[df_1.index[0], 'theta_group']):
                        print(f"3 adding edge: {[df_0.index[-1], df_1.index[0]]}")
                        edges.append([df_0.index[-1], df_1.index[0]])
                elif df_1.loc[df_1.index[-1]]['z'] > df_0.loc[df_0.index[0]]['z'] and np.abs(df_1.loc[df_1.index[-1], 'r_cyl'] - df_0.loc[df_0.index[0], 'r_cyl']) < 20:
                    if not (df_1.loc[df_1.index[-1], 'phi_group'] != df_0.loc[df_0.index[0], 'phi_group'] and df_1.loc[df_1.index[-1], 'theta_group'] != df_0.loc[df_0.index[0], 'theta_group']):
                        print(f"4 adding edge: {[df_1.index[-1], df_0.index[0]]}")
                        edges.append([df_1.index[-1], df_0.index[0]])
                    
            elif df_0['theta_group'].nunique(0) == 1:
                if df_0.loc[df_0.index[-1]]['z'] > df_1.loc[df_1.index[0]]['z']:
                    print(f"5 adding edge: {[df_0.index[-1], df_1.index[0]]}")
                    edges.append([df_0.index[-1], df_1.index[0]])
                elif df_1.loc[df_1.index[-1]]['z'] > df_0.loc[df_0.index[0]]['z']:
                    print(f"6 adding edge: {[df_1.index[-1], df_0.index[0]]}")
                    edges.append([df_1.index[-1], df_0.index[0]])
                    
            if np.min(df_0['z']) > np.max(df_1['z']):
                for leaf in df_1.index:
                    if df_1['phi_group'].nunique(0) == 1 and df_1['theta_group'].nunique(0) > 1:
                        if [df_0.index[-1], leaf] in edges: continue
                        print(f"7 adding edge: {[df_0.index[-1], leaf]}")
                        edges.append([df_0.index[-1], leaf]) 
            else:
                for leaf in df_0.index:
                    if df_0['phi_group'].nunique(0) == 1 and df_0['theta_group'].nunique(0) > 1:
                        if [df_1.index[-1], leaf] in edges: continue
                        print(f"8 adding edge: {[df_1.index[-1], leaf]}")
                        edges.append([df_1.index[-1], leaf]) 
    
            
            df_sub_branch = df_pat.loc[compare[0] + compare[1]]
            if df_sub_branch['phi_group'].nunique(0) == 1:
                df_sub_branch_z = df_sub_branch.sort_values('z', ascending=False)
                for i, gtv in enumerate(df_sub_branch_z.index):
                    if i >= len(df_sub_branch_z)-1: continue
                    gtv2 = df_sub_branch_z.index[i+1]
                    if [gtv, gtv2] in edges: continue
                    print(f"8.111 adding edge {[gtv, gtv2]}")
                    edges.append([gtv, gtv2])
            #df_sub_branch.sort_values(['phi_group', 'r'], inplace=True)
            df_sub_branch.sort_values(['phi_group', 'r'], inplace=True)
    
            if df_sub_branch['phi_group'].nunique(0) == len(df_sub_branch):
                for gtv in df_sub_branch.index:
                    if gtv not in skip_to_primary:
                        skip_to_primary.append(gtv)
            elif df_sub_branch['phi_group'].nunique(0) >  1:
                phi_counts = df_sub_branch.value_counts('phi_group')
                min_phi_count = phi_counts.min()
                for phi in phi_counts.index:
                    if phi_counts.loc[phi] != min_phi_count: continue
                    gtv = df_sub_branch[df_sub_branch['phi_group']==phi].index[0]
                    if gtv not in skip_to_primary:
                        print(f"8.11 adding to skip to primary: {gtv}")
                        skip_to_primary.append(gtv)
                #group_to_send = phi_counts[phi_counts == phi_counts.min()].index[0]
                #skip_to_primary += list(df_sub_branch[df_sub_branch['phi_group']==group_to_send].index)
            if (df_sub_branch.loc[:, 'r_cyl'].max() - df_sub_branch.loc[:, 'r_cyl'].min() < 10) and (df_sub_branch.loc[:, 'z'].max() - df_sub_branch.loc[:, 'z'].min() < 15):
                df_sub_branch.sort_values('z', ascending=False, inplace=True)
                for i, gtv in enumerate(df_sub_branch.index):
                    if i >= len(df_sub_branch)-1: continue
                    gtv2 = df_sub_branch.index[i+1]
                    if [gtv, gtv2] in edges: continue
                    print(f"8.12 adding edge {[gtv, gtv2]}")
                    edges.append([gtv, gtv2]) 
            return list(df_sub_branch.index[:2])    
    
        to_compare = []
        leaves = connections[branch]
        for leaf in leaves:
            print(f"leaf: {leaf}")
            if leaf in connections.keys():
                print(f"    entering recursion for {leaf}")
                to_compare.append(recursive_walk(leaf, skip_to_primary))
            else:
                to_compare.append([leaf])
    
        print(f"done with recursion, for branch: {branch}")
        last_compare = []

        print(to_compare)
        if len(to_compare) == 1:
            return to_compare[0]
 
        if len(to_compare) == 2 and isinstance(to_compare[0], str) and isinstance(to_compare[1], str):
            df_c = df_pat.loc[to_compare]
            df_c.sort_values('z', ascending=False, inplace=True)
            if [df_c.index[0], df_c.index[1]] not in edges:
                print(f"8.1 adding edge {[df_c.index[0], df_c.index[1]]}")
                edges.append([df_c.index[0], df_c.index[1]])
            return to_compare
        for i, leaf in enumerate(to_compare):
            if isinstance(leaf, str):
                to_compare[i] = [leaf]
        print(to_compare)
        df_0 = df_pat.loc[to_compare[0]]
        df_0.sort_values('z', ascending=False, inplace=True)
        df_1 = df_pat.loc[to_compare[1]]
        df_1.sort_values('z', ascending=False, inplace=True)
   
        df_comp = df_pat.loc[to_compare[0] + to_compare[1]]
        #df_comp.sort_values(['phi_group', 'theta_group', 'r'], inplace=True)
        #df_comp.sort_values(['theta_group', 'r'], inplace=True)
        df_comp.sort_values(['z'], ascending=False, inplace=True)
        print(df_comp)
        for i, gtv in enumerate(df_comp.index):
            if i == len(df_comp.index)-1: continue
            gtv2 = df_comp.index[i+1]
            if df_comp.loc[gtv, 'phi_group'] != df_comp.loc[gtv2, 'phi_group'] and df_comp.loc[gtv, 'theta_group'] != df_comp.loc[gtv2, 'theta_group']: continue
            if np.abs(df_comp.loc[gtv, 'phi'] - df_comp.loc[gtv2, 'phi']) < 0.5 and df_comp.loc[gtv, 'r_cyl'] - df_comp.loc[gtv2, 'r_cyl'] > 20:
                if np.abs(df_comp.loc[gtv, 'z'] - df_comp.loc[gtv2, 'z']) > 10: continue
                if [gtv2, gtv] in edges: continue
                print(f"9 adding edge: {[gtv2, gtv]}")
                edges.append([gtv2, gtv])
            else:
                if [gtv, gtv2] in edges: continue
                print(f"9.1 adding edge: {[gtv, gtv2]}")
                edges.append([gtv, gtv2])
               
        if len(df_0) == 2 and df_0['phi_group'].nunique(0) == 1:
            if list(df_0.index) not in edges:
                print(f"10 adding edge: {list(df_0.index)}")
                edges.append(list(df_0.index))
           
        if len(df_1) == 2 and df_1['phi_group'].nunique(0) == 1:
            if list(df_1.index) not in edges:
                print(f"11 adding edge: {list(df_1.index)}")
                edges.append(list(df_1.index))
 
        if np.min(df_0['z']) > np.max(df_1['z']):
            for leaf in df_1.index:
                if df_1.loc[leaf, 'phi_group'] == df_0.loc[df_0.index[-1], 'phi_group'] and df_1.loc[leaf, 'theta_group'] != df_0.loc[df_0.index[-1], 'theta_group']:
                    if len(df_0) == 2 and abs(df_0.iloc[0]['theta'] - df_0.iloc[1]['theta']) < 1.0:
                        if np.abs(df_0.loc[df_0.index[0], 'phi'] - df_0.loc[df_0.index[1], 'phi']) > 0.5:
                            for gtv in df_0.index:
                                if df_1.loc[leaf, 'z'] > df_0.loc[gtv, 'z']: continue
                                if [gtv, leaf] in edges: continue
                                print("*****************************************")
                                print(f"12 adding edge: {[gtv, leaf]}")
                                print("*****************************************")
                                edges.append([gtv, leaf])
                        else:
                            if df_1.loc[leaf, 'z'] > df_0.loc[df_0.index[-1], 'z']: continue
                            if [df_0.index[-1], leaf] in edges: continue
                            print("*****************************************")
                            print(f"12.1 adding edge: {[df_0.index[-1], leaf]}")
                            print("*****************************************")
                            edges.append([df_0.index[-1], leaf])
                    else:
                        if df_1.loc[leaf, 'z'] > df_0.loc[df_0.index[-1], 'z']: continue
                        if [df_0.index[-1], leaf] in edges: continue
                        print("*****************************************")
                        print(f"13 adding edge: {[df_0.index[-1], leaf]}")
                        print("*****************************************")
                        edges.append([df_0.index[-1], leaf]) 
                elif df_1.loc[leaf, 'phi_group'] == df_0.loc[df_0.index[-1], 'phi_group'] and df_1.loc[leaf, 'theta_group'] == df_0.loc[df_0.index[-1], 'theta_group']:
                    if [df_0.index[-1], leaf] in edges: continue
                    print("*****************************************")
                    print(f"14 adding edge: {[df_0.index[-1], leaf]}")
                    print("*****************************************")
                    edges.append([df_0.index[-1], leaf]) 
                   
        else:
            for leaf in df_0.index:
                if df_0.loc[leaf, 'phi_group'] == df_1.loc[df_1.index[-1], 'phi_group'] and df_0.loc[leaf, 'theta_group'] != df_1.loc[df_1.index[-1], 'theta_group']:
                    if len(df_1) == 2 and abs(df_1.iloc[0]['theta'] - df_1.iloc[1]['theta']) < 1.0:
                        for gtv in df_1.index:
                            if df_0.loc[leaf, 'z'] > df_1.loc[gtv, 'z']: continue
                            if [gtv, leaf] in edges: continue
                            print("*****************************************")
                            print(f"15 adding edge: {[gtv, leaf]}")
                            print("*****************************************")
                            edges.append([gtv, leaf]) 
                    else:
                        if df_0.loc[leaf, 'z'] > df_1.loc[df_1.index[-1], 'z']: continue
                        if [df_1.index[-1], leaf] in edges: continue
                        print("else******************************************")
                        print(f"16 adding edge: {[df_1.index[-1], leaf]}")
                        print("*****************************************")
                        edges.append([df_1.index[-1], leaf]) 
       
        df_branch = df_pat.loc[to_compare[0] + to_compare[1]]
        df_branch.sort_values('r', inplace=True)
        df_branch_z = df_branch.sort_values(['phi_group', 'z'], ascending=[True, False])
        for i, gtv in enumerate(df_branch_z.index):
            if i >= len(df_branch_z)-1: continue
            gtv2 = df_branch_z.index[i+1]
            if df_branch_z.loc[gtv, 'phi_group'] == df_branch_z.loc[gtv2, 'phi_group']:
                if [gtv, gtv2] in edges: continue
                print(f"16.1 adding edge {[gtv, gtv2]}")
                edges.append([gtv, gtv2])
        if df_branch['phi_group'].nunique(0) == len(df_branch):
            for gtv in df_branch.index:
                if gtv not in skip_to_primary:
                    skip_to_primary.append(gtv)
        elif df_branch['phi_group'].nunique(0) >  1:
            phi_counts = df_branch.value_counts('phi_group')
            min_phi_count = phi_counts.min()
            for phi in phi_counts.index:
                if phi_counts.loc[phi] != min_phi_count: continue
                gtv = df_branch[df_branch['phi_group'] == phi].index[0]
                if gtv not in skip_to_primary:
                    print(f"17 adding to skip to primary: {gtv}")
                    skip_to_primary.append(gtv)
            #group_to_send = phi_counts[phi_counts == phi_counts.min()].index[0]
            #print(f"17 adding to skip to primary: {list(df_branch[df_branch['phi_group']==group_to_send].index)}")
            #print(group_to_send)
            #group_to_send_gtvs = list(df_branch[df_branch['phi_group']==group_to_send].index)

            #for gtv in group_to_send_gtvs:
            #    if gtv not in skip_to_primary:
            #        skip_to_primary.append(gtv)

            front_to_send = phi_counts[phi_counts == phi_counts.max()].index[0]
            print(f"18 adding to skip to primary: {list(df_branch[df_branch[df_branch['phi_group']==front_to_send]['r_cyl'].min() == df_branch['r_cyl']].index)}")
            front_to_send_gtvs = list(df_branch[df_branch[df_branch['phi_group']==front_to_send]['r_cyl'].min() == df_branch['r_cyl']].index)
            print(front_to_send_gtvs)
            for gtv in front_to_send_gtvs:
                if gtv not in skip_to_primary:
                    skip_to_primary.append(gtv)
        for gtv in df_branch.index:
            if df_branch.loc[gtv, 'z'] > 0:
                skip_to_primary.append(gtv)
        return list(df_branch.index[:2])
    
    
    final_compare = []
    for leaf in connections['p']:
        final_compare.append(process_tree(leaf, skip_to_primary))
    print(f"finished tree recursion, starting final comparisons")
    last_edges = []
    print(final_compare)
    df_f = df_pat.loc[final_compare[0] + final_compare[1]]
    df_f.sort_values('r', inplace=True)
    print(df_f)
    #df_f.sort_values('z', ascending=False, inplace=True)
    if df_f['phi_group'].nunique(0) == 1:
        if df_f['theta_group'].nunique(0) == 1:
            df_f_z = df_f.sort_values('z', ascending=False)
            for i, gtv in enumerate(df_f_z.index):
                if i == len(df_f_z.index)-1: continue
                gtv2 = df_f_z.index[i+1]
                if np.abs(df_f_z.loc[gtv, 'z'] - df_f_z.loc[gtv2, 'z']) < 5:
                    df_f_rcyl = df_f_z.loc[[gtv, gtv2]].sort_values('r_cyl')
                    if [df_f_rcyl.index[0], df_f_rcyl.index[1]] not in edges:
                        print(f"source branch is all the same phi group, adding extra edge between branches")
                        print(f"19 adding edge: {[df_f_rcyl.index[0], df_f_rcyl.index[1]]}")
                        edges.append([df_f_rcyl.index[0], df_f_rcyl.index[1]])
                else:
                    if [df_f_z.index[i], df_f_z.index[i+1]] not in edges:
                        print(f"source branch is all the same phi group, adding extra edge between branches")
                        print(f"19.1 adding edge: {[df_f_z.index[i], df_f_z.index[i+1]]}")
                        edges.append([df_f_z.index[i], df_f_z.index[i+1]])
        else:
            df_f.sort_values(['z'], ascending=False, inplace=True)
            for i, gtv in enumerate(df_f.index):
                if i == len(df_f.index)-1: continue
                if [df_f.index[i], df_f.index[i+1]] not in edges:
                    print(f"source branch is all the same phi group, adding extra edge between branches")
                    print(f"source branch has different theta groups, reordering first connection by z")
                    print(f"20 adding edge: {[df_f.index[i], df_f.index[i+1]]}")
                    edges.append([df_f.index[i], df_f.index[i+1]])

            df_f0 = df_pat.loc[final_compare[0]]
            df_f0.sort_values('z', ascending=False, inplace=True) 
            df_f1 = df_pat.loc[final_compare[1]] 
            df_f1.sort_values('z', ascending=False, inplace=True)
            if np.min(df_f0['z']) > np.max(df_f1['z']):
                if len(df_f0) == 2:
                    for leaf in df_f1.index:
                        for gtv in df_f0.index:
                            if df_f1.loc[leaf, 'z'] > df_f0.loc[gtv, 'z']: continue
                            if np.abs(df_f1.loc[leaf, 'theta'] - df_f0.loc[gtv, 'theta']) > 0.75: continue
                            if [gtv, leaf] in edges: continue
                            print("*****************************************")
                            print(f"21 adding edge: {[gtv, leaf]}")
                            print("*****************************************")
                            edges.append([gtv, leaf])
                else:
                    for leaf in df_f1.index:
                            gtv = df_f0.index[0]
                            if df_f1.loc[leaf, 'z'] > df_f0.loc[gtv, 'z']: continue
                            if np.abs(df_f1.loc[leaf, 'theta'] - df_f0.loc[gtv, 'theta']) > 0.75: continue
                            if [gtv, leaf] in edges: continue
                            print("*****************************************")
                            print(f"22 adding edge: {[gtv, leaf]}")
                            print("*****************************************")
                            edges.append([gtv, leaf])
                  
            else:
                if len(df_f1) == 2:
                    for leaf in df_f0.index:
                        for gtv in df_f1.index:
                            if df_f0.loc[leaf, 'z'] > df_f1.loc[gtv, 'z']: continue
                            if np.abs(df_f0.loc[leaf, 'theta'] - df_f1.loc[gtv, 'theta']) > 0.75: continue
                            if [gtv, leaf] in edges: continue
                            print("else*****************************************")
                            print(f"23 adding edge: {[gtv, leaf]}")
                            print("*****************************************")
                            edges.append([gtv, leaf]) 
                else:
                    for leaf in df_f0.index:
                            gtv = df_f1.index[0]
                            if df_f0.loc[leaf, 'z'] > df_f1.loc[gtv, 'z']: continue
                            if np.abs(df_f0.loc[leaf, 'theta'] - df_f1.loc[gtv, 'theta']) > 0.75: continue
                            if [gtv, leaf] in edges: continue
                            print("*****************************************")
                            print(f"23.1 adding edge: {[gtv, leaf]}")
                            print("*****************************************")
                            edges.append([gtv, leaf])
    else:
        df_f.sort_values('z', ascending=False, inplace=True)
        for i, gtv in enumerate(df_f.index):
            for j, gtv2 in enumerate(df_f.index):
                if i >= j: continue
                if df_f.loc[gtv, 'phi_group'] == df_f.loc[gtv2, 'phi_group']:
                    if [gtv, gtv2] in edges: continue
                    print(f"24 adding edge: {[gtv, gtv2]}")
                    edges.append([gtv, gtv2])
                if df_f.loc[gtv, 'theta_group'] == df_f.loc[gtv2, 'theta_group'] and np.abs(df_f.loc[gtv, 'phi'] - df_f.loc[gtv2, 'phi']) < 0.5:
                    if [gtv, gtv2] in edges: continue
                    print(f"24.11 adding edge: {[gtv, gtv2]}")
                    edges.append([gtv, gtv2])
        df_f0 = df_pat.loc[final_compare[0]]
        df_f0.sort_values('z', ascending=False, inplace=True) 
        df_f1 = df_pat.loc[final_compare[1]] 
        df_f1.sort_values('z', ascending=False, inplace=True)

        if len(df_f0) == 2 and (df_f0.iloc[0]['phi'] - df_f0.iloc[1]['phi'] < 1.0): 
            if [df_f0.index[0], df_f0.index[1]] not in edges:
                print(f"24.12 adding edge: {[df_f0.index[0], df_f0.index[1]]}")
                edges.append([df_f0.index[0], df_f0.index[1]])
        if len(df_f1) == 2 and (df_f1.iloc[0]['phi'] - df_f1.iloc[1]['phi'] < 1.0): 
            if [df_f1.index[0], df_f1.index[1]] not in edges:
                print(f"24.12 adding edge: {[df_f1.index[0], df_f1.index[1]]}")
                edges.append([df_f1.index[0], df_f1.index[1]])

        if np.min(df_f0['z']) > np.max(df_f1['z']):
            if len(df_f0) == 2 and abs(df_f0.iloc[0]['theta'] - df_f0.iloc[1]['theta']) < 1.0:
                for leaf in df_f1.index:
                    for gtv in df_f0.index:
                        if df_f1.loc[leaf, 'z'] > df_f0.loc[gtv, 'z']: continue
                        if np.abs(df_f1.loc[leaf, 'r_cyl'] - df_f0.loc[gtv, 'r_cyl']) > 20: continue
                        if np.abs(df_f1.loc[leaf, 'phi'] - df_f0.loc[gtv, 'phi']) > 1.0: continue
                     
                        if [gtv, leaf] in edges: continue
                        print("*****************************************")
                        print(f"24.1 adding edge: {[gtv, leaf]}")
                        print("*****************************************")
                        edges.append([gtv, leaf])
            else:
                for leaf in df_f1.index:
                    gtv = df_f0.index[0]
                    if df_f1.loc[leaf, 'z'] > df_f0.loc[gtv, 'z']: continue
                    if np.abs(df_f1.loc[leaf, 'r_cyl'] - df_f0.loc[gtv, 'r_cyl']) > 20: continue
                    if np.abs(df_f1.loc[leaf, 'phi'] - df_f0.loc[gtv, 'phi']) > 1.0: continue
                    if [gtv, leaf] in edges: continue
                    print("*****************************************")
                    print(f"24.2 adding edge: {[gtv, leaf]}")
                    print("*****************************************")
                    edges.append([gtv, leaf])
        else:
            if len(df_f1) == 2 and abs(df_f1.iloc[0]['theta'] - df_f1.iloc[1]['theta']) < 1.0:
                for leaf in df_f0.index:
                    for gtv in df_f1.index:
                        if df_f0.loc[leaf, 'z'] > df_f1.loc[gtv, 'z']: continue
                        if np.abs(df_f0.loc[leaf, 'r_cyl'] - df_f1.loc[gtv, 'r_cyl']) > 20: continue
                        if np.abs(df_f0.loc[leaf, 'phi'] - df_f1.loc[gtv, 'phi']) > 1.0: continue
                        if [gtv, leaf] in edges: continue
                        print("else*****************************************")
                        print(f"24.3 adding edge: {[gtv, leaf]}")
                        print("*****************************************")
                        edges.append([gtv, leaf]) 
            else:
                for leaf in df_f0.index:
                        gtv = df_f1.index[0]
                        if df_f0.loc[leaf, 'z'] > df_f1.loc[gtv, 'z']: continue
                        if np.abs(df_f0.loc[leaf, 'r_cyl'] - df_f1.loc[gtv, 'r_cyl']) > 20: continue
                        if np.abs(df_f0.loc[leaf, 'phi'] - df_f1.loc[gtv, 'phi']) > 1.0: continue
                        if [gtv, leaf] in edges: continue
                        print("*****************************************")
                        print(f"24.4 adding edge: {[gtv, leaf]}")
                        print("*****************************************")
                        edges.append([gtv, leaf])
 

                    
                 
    for c in final_compare:
        df_c = df_pat.loc[c]
        print(df_c)
        #df_c.sort_values('r', inplace=True)
        if len(df_c) == 1:
            last_edges += list(df_c.index)
            continue
        df_c.sort_values('z', ascending=False, inplace=True)
        if df_c['phi_group'].nunique(0) == 1 and df_c['theta_group'].nunique(0) == 1:
            if (df_c.iloc[0]['r_cyl'] - df_c.iloc[1]['r_cyl']) < 0:
                last_edges += [df_c.index[0]]
            else:
                last_edges += list(df_c.index)
        else:
            last_edges += list(df_c.index)
    print(f"entries chosen for connection to primary: {last_edges}")
    print(skip_to_primary)
    for gtv in last_edges:
        skip_connection = False
        for gtv_skip in skip_to_primary:
            if gtv == gtv_skip: continue
            df_c = df_pat.loc[[gtv, gtv_skip]]
            if df_c['phi_group'].nunique(0) == 1:
                df_c.sort_values('z', ascending=False, inplace=True)
                if [df_c.index[0], df_c.index[1]] not in edges: 
                    print(f"skip to primary has cross-branch entry, making connections to opposite branch")
                    print(f"25 adding edge: {[df_c.index[0], df_c.index[1]]}")
                    edges.append([df_c.index[0], df_c.index[1]])
                if df_c.loc[df_c.index[0], 'z'] > 0:
                    skip_connection = False
                else:
                    skip_connection = True
        if not skip_connection:
            print(f"making final connections to primary") 
            print(f"26 adding edge: {[primary, gtv]}")
            edges.append([primary, gtv])
        else:
            print(f"{gtv} superceded by skip entry, dropping connection to primary") 
    skip_connection = []
    for gtv in skip_to_primary:
        for gtv2 in skip_to_primary:
            if gtv == gtv2: continue
            df_c = df_pat.loc[[gtv, gtv2]]
            df_c.sort_values('z', ascending=False, inplace=True)
            if df_c['phi_group'].nunique(0) == 1:
                if [df_c.index[0], df_c.index[1]] not in edges:
                    print(f"skip to primary has cross-branch entry, making connections to opposite branch")
                    print(f"27 adding edge: {[df_c.index[0], df_c.index[1]]}")
                    edges.append([df_c.index[0], df_c.index[1]])
                skip_connection.append(df_c.index[1])
        if gtv in skip_connection: 
            print(f"{gtv} superceded by skip entry, dropping connection to primary")
        else: 
            print(f"connecting skip to primary entries")
            if [primary, gtv] in edges: continue
            print(f"28 adding edge: {[primary, gtv]}")
            edges.append([primary,gtv])

    return edges
