import os
import glob
import numpy as np
import networkx as nx
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import datasets

def draw_a_graph(filename, dataset, topk_all=None, topk_per_step=None, font_size=4, node_size=100, edge_width=0.5, disable_draw=False):
    nodes_per_step = []
    rels_dct = {}
    edges_set = {}
    head, relation, tail = os.path.basename(filename)[len('test_epoch1_'):].split('.')[0].split('->')
    head = head.split('(')[0]
    relation = relation.split('(')[0]
    tail = tail.split('(')[0]
    
    with open(filename) as fin:
        mode = None
        for line in fin.readlines():
            line = line.strip()
            if line == 'nodes:':
                mode = 'nodes'
            elif line == 'edges:':
                mode = 'edges'
            else:
                if mode == 'nodes':
                    nodes = []
                    max_att = 0.
                    for sp in line.split('\t'):
                        sp2 = sp.split(':')
                        node_att = float(sp2[1])
                        max_att = max(max_att, node_att)
                        sp2 = sp2[0].split('(')
                        node_id = sp2[0]
                        node_name = sp2[1][:-1]
                        nodes.append((node_id, node_name, node_att))

                    if topk_per_step is not None:
                        nodes = sorted(nodes, key=lambda node: -node[2])[:topk_per_step]

                    nodes = {node_id: (node_name, node_att / max_att) for node_id, node_name, node_att in nodes}
                    nodes_per_step.append(nodes)

                elif mode == 'edges':
                    edges = []
                    for sp in line.split('\t'):
                        sp2 = sp.split('->')
                        node_id1 = sp2[0].split('(')[0]
                        node_id2 = sp2[2].split('(')[0]
                        rel_id = sp2[1].split('(')[0]
                        rel_name = sp2[1].split('(')[1][:-1]
                        rels_dct[rel_id] = rel_name
                        edges.append((node_id1, rel_id, node_id2))
                        if (node_id1, rel_id, node_id2) not in edges_set:
                            edges_set[(node_id1, rel_id, node_id2)] = len(edges_set)

    nodes = {}
    n_steps = len(nodes_per_step)
    i = 0
    for t in range(n_steps):
        for k, v in nodes_per_step[t].items():
            node_id = k
            node_name = v[0]
            node_att = v[1]
            att_h = node_att * np.power(0.9, t)
            att_t = node_att * np.power(0.9, n_steps-1-t)

            att = 0.5 - att_h / 2 if att_h > att_t else 0.5 + att_t / 2
            node_att = nodes[node_id][1] if node_id in nodes else 0.5
            nodes[node_id] = (node_name, att) if abs(att - 0.5) > abs(node_att - 0.5) else (node_name, node_att)

    nodes_all = [(i, {'id': k, 'name': v[0], 'att': v[1]}) for i, (k, v) in enumerate(nodes.items())]

    if topk_all is not None:
        nodes_all = sorted(nodes_all, key=lambda node: -node[1]['att'])[:topk_all]
        nodes_all = [(i, e[1]) for i, e in enumerate(nodes_all)]

    id2i = {node[1]['id']: node[0] for node in nodes_all}
    i2id = {node[0]: node[1]['id'] for node in nodes_all}

    edges = list(edges_set.keys())
    edges = sorted(edges, key=lambda e: edges_set[e])
    edges_all = [(id2i[n1], id2i[n2], {'rel_id': r, 'rel_name': rels_dct[r]})
                 for n1, r, n2 in edges if n1 in id2i and n2 in id2i]

    fig = None
    if not disable_draw:
        graph = nx.MultiGraph()
        graph.add_nodes_from(nodes_all)
        graph.add_edges_from(edges_all)
        
        # 使用spring_layout代替graphviz_layout以提高兼容性
        pos = nx.spring_layout(graph, seed=42)  
        
        # 创建节点和边的轨迹
        node_x = []
        node_y = []
        node_text = []
        node_att = []
        node_ids = []
        node_sizes = []
        
        for node in nodes_all:
            idx = node[0]
            x, y = pos[idx]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node[1]['name']}<br>ID: {node[1]['id']}")
            node_att.append(node[1]['att'])
            node_ids.append(node[1]['id'])
            
            # 根据节点重要性调整大小
            base_size = node_size * 10
            if node[1]['id'] == head or node[1]['id'] == tail:
                size = base_size * 2.5
            else:
                size = base_size * (1 + node[1]['att'] * 0.5)
            node_sizes.append(size)
        
        # 创建节点轨迹
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='top center',
            textfont=dict(size=font_size),
            marker=dict(
                showscale=True,
                colorscale='Portland',
                color=node_att,
                size=node_sizes,
                colorbar=dict(
                    thickness=15,
                    title='Node Attention',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=1,
                line_color='yellow'
            ),
            hoverinfo='text',
            name='Nodes'
        )
        
        # 创建边轨迹
        edge_x = []
        edge_y = []
        edge_att = []
        edge_text = []
        
        for edge in edges_all:
            start_idx, end_idx, attr = edge
            x0, y0 = pos[start_idx]
            x1, y1 = pos[end_idx]
            
            # 添加起点
            edge_x.append(x0)
            edge_y.append(y0)
            edge_att.append(0.5)
            edge_text.append('')
            
            # 添加中间点（用于弯曲效果）
            edge_x.append((x0 + x1) / 2)
            edge_y.append((y0 + y1) / 2)
            edge_att.append(1.0)
            edge_text.append(attr['rel_name'])
            
            # 添加终点
            edge_x.append(x1)
            edge_y.append(y1)
            edge_att.append(0.5)
            edge_text.append('')
            
            # 添加None分隔每条线
            edge_x.append(None)
            edge_y.append(None)
            edge_att.append(None)
            edge_text.append(None)
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=edge_width * 2, color='#888'),
            hoverinfo='text',
            text=edge_text,
            mode='lines+text',
            textposition='middle center',
            textfont=dict(size=font_size),
            name='Edges'
        )
        
        # 创建图形
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=f'Graph: {dataset.id2entity[int(head)]} -> {dataset.id2relation[int(relation)]} -> {dataset.id2entity[int(tail)]}',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            height=600
                        ))
    
    edges = [(i2id[e[0]], e[2]['rel_id'], i2id[e[1]]) for e in edges_all]
    return head, relation, tail, edges, fig


def draw(dataset, dirpath, new_dirpath):
    if not os.path.exists(new_dirpath):
        os.mkdir(new_dirpath)

    for filename in glob.glob(os.path.join(dirpath, '*.txt')):
        try:
            print(filename)
            # 创建包含两个子图的Plotly图形
            fig = make_subplots(rows=1, cols=2, 
                               subplot_titles=('Full Graph', 'Top-k Filtered Graph'),
                               horizontal_spacing=0.05)
            
            # 绘制完整图形
            head, rel, tail, edges, fig1 = draw_a_graph(filename, dataset, 
                                                      font_size=3)
            
            # 绘制过滤后的图形
            _, _, _, _, fig2 = draw_a_graph(filename, dataset, 
                                          topk_per_step=5, 
                                          font_size=5, 
                                          node_size=180, 
                                          edge_width=1)
            
            # 添加两个子图到主图
            for trace in fig1.data:
                fig.add_trace(trace, row=1, col=1)
                
            for trace in fig2.data:
                fig.add_trace(trace, row=1, col=2)
            
            # 更新布局
            fig.update_layout(
                height=600,
                showlegend=False,
                title_text=f'Graph Comparison: {dataset.id2entity[int(head)]} -> {dataset.id2relation[int(rel)]} -> {dataset.id2entity[int(tail)]}',
                title_x=0.5
            )
            
            # 保存为HTML文件
            html_filename = os.path.join(new_dirpath, os.path.basename(filename)[:-4] + '.html')
            fig.write_html(html_filename)
            print(f"Saved visualization to {html_filename}")
            
            # 保存文本信息
            with open(os.path.join(new_dirpath, os.path.basename(filename)), 'w') as fout:
                fout.write('{}\t{}\t{}\n\n'.format(dataset.id2entity[int(head)],
                                                   dataset.id2relation[int(rel)],
                                                   dataset.id2entity[int(tail)]))
                for h, r, t in edges:
                    fout.write('{}\t{}\t{}\n'.format(dataset.id2entity[int(h)],
                                                     dataset.id2relation[int(r)],
                                                     dataset.id2entity[int(t)]))
        except IndexError:
            print('Cause `IndexError` for file `{}`'.format(filename))
        except Exception as e:
            print(f'Error processing file {filename}: {str(e)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None,
                        choices=['FB237', 'FB237_v2', 'FB15K', 'WN18RR', 'WN18RR_v2', 'WN', 'YAGO310', 'NELL995'])
    args = parser.parse_args()

    ds = getattr(datasets, args.dataset)()
    if args.dataset == 'NELL995':
        nell995_cls = getattr(datasets, args.dataset)
        for ds in nell995_cls.datasets():
            print('nell > ' + ds.name)
            dir_name = '../output/NELL995_subgraph/' + ds.name
            if not os.path.exists(dir_name):
                continue
            dir_name_2 = '../visual/NELL995_subgraph/' + ds.name
            os.makedirs(dir_name_2, exist_ok=True)
            draw(ds, dir_name, dir_name_2)
    else:
        ds = getattr(datasets, args.dataset)()
        print(ds.name)
        dir_name = '../output/' + ds.name + '_subgraph'
        dir_name_2 = '../visual/' + ds.name + '_subgraph'
        os.makedirs(dir_name_2, exist_ok=True)
        draw(ds, dir_name, dir_name_2)