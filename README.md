# NeuCFlow
Neural Consciousness Flow

## Training and Evaluating

```bash
./run.sh --dataset <Dataset>
```

<Dataset> can be one of 'FB237', 'FB237_v2', 'FB15K', 'WN18RR', 'WN18RR_v2', 'WN', 'YAGO310', 'NELL995'.

## Visualization

Run with `test_output_attention` to get data files of extracted subgraphs for each query in test. For example, if you want to get data files on the NELL995 dataset (containing several separate datasets), with `max_attended_nodes=20`, run:

```bash
./run.sh --dataset NELL995 --test_output_attention --max_attended_nodes 20 --test_max_attended_nodes 20
```

Then, you get data files in the `output/NELL995_subgraph` directory. Next, visualize them by:

```bash
cd code
python visualize.py --dataset NELL995
```

You will find image files for visualization in the `visual/NELL995_subgraph` directory.

NeuCFlow

神经意识流 (Neural Consciousness Flow)

训练与评估

./run.sh --dataset <数据集名称>
$env:CUDA_VISIBLE_DEVICES=0; python run.py --dataset

<数据集名称> 可选 'FB237', 'FB237_v2', 'FB15K', 'WN18RR', 'WN18RR_v2', 'WN', 'YAGO310', 'NELL995' 中的任意一个。

可视化

运行测试时添加 test_output_attention 参数可为每个测试查询提取子图数据文件。例如，若需在 NELL995 数据okookokooo集（包含多个子数据集）上获取数据文件，并设置 max_attended_nodes=20，请运行：
./run.sh --dataset NELL995 --test_output_attention --max_attended_nodes 20 --test_max_attended_nodes 20



运行后，您将在 output/NELL995_subgraph 目录中找到生成的数据文件。接着通过以下命令进行可视化：
cd code
python visualize.py --dataset NELL995


可视化结果图像文件将保存在 visual/NELL995_subgraph 目录中。
ganjue