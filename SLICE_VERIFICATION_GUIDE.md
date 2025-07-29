# 切片加载验证指南

## 如何验证每个节点只加载自己分配的切片

### 方法1：实时日志验证（需要DEBUG>10）
设置环境变量启用详细日志：
```bash
export DEBUG=11
```

运行exo时，你会看到以下详细日志输出：

#### 🎯 分区分配验证
```
🎯 === NODE SLICE ASSIGNMENT VERIFICATION ===
🖥️  Node ID: node1
📊 Model: llama-3b
🧩 Total layers: 24
🔗 Partition index: 0
👥 Active nodes: 3

📋 COMPLETE PARTITION MAP:
   ✅ Node 0: node1
      Layers: 0-9 (10 layers, 41.7%)
      Memory weight: [0.000, 0.417)
   ⏳ Node 1: node2
      Layers: 10-19 (10 layers, 41.7%)
      Memory weight: [0.417, 0.833)
   ⏳ Node 2: node3
      Layers: 20-23 (4 layers, 16.7%)
      Memory weight: [0.833, 1.000)

🎯 THIS NODE WILL LOAD:
   Model: llama-3b
   Layers: 0-9
   Layer count: 10 / 24
   Percentage: 41.7%
   Memory savings: 58.3% reduction per node
```

#### 📦 文件下载验证
```
📦 === SLICE LOADING VERIFICATION ===
🎯 Shard: llama-3b layers 0-9
🖥️  Node: mlx
📊 Layer range: 0-9 (10 layers)
🎯 Allow patterns: ['model-00001-of-00003.safetensors', ...]

📥 === DOWNLOAD VERIFICATION ===
📁 Total files in repo: 45
🎯 Files after slice filtering: 3
💾 Full repo size: 6174.50 MB
💾 This slice size: 2058.17 MB
🎯 Size reduction: 66.7%

📋 FILES FOR THIS SLICE:
    1. 📄 model-00001-of-00003.safetensors (2058.17 MB)
    2. 📄 tokenizer.model (0.49 MB)
    3. 📄 config.json (0.01 MB)

✅ === DOWNLOAD COMPLETED VERIFICATION ===
🎯 Shard: llama-3b layers 0-9
📁 Files downloaded: 3
   📄 model-00001-of-00003.safetensors: 2058.17 MB ✅
   📄 tokenizer.model: 0.49 MB ✅
   📄 config.json: 0.01 MB ✅

💾 DOWNLOAD SUMMARY:
   Expected: 2058.67 MB
   Actual:   2058.67 MB
   Status:   ✅ Complete
```

#### 🚀 运行时验证
```
🚀 === RUNTIME SLICE VERIFICATION ===
🖥️  Node: node1
📊 Model: llama-3b
🧩 Loaded layers: 0-9
🎯 Layer count: 10 / 24
```

### 方法2：使用验证脚本
```bash
# 运行验证脚本
python verify_slice_loading.py
```

### 方法3：手动检查

#### 1. 检查下载目录
每个节点的下载目录只包含其分配的权重文件：
```bash
# 检查node1的下载目录
ls ~/.cache/exo/downloads/models--llama-3b/
# 应该只看到部分safetensors文件，而不是全部

# 对比完整模型大小
du -sh ~/.cache/exo/downloads/models--llama-3b/
# node1: 2.1G (部分)
# 完整: 6.2G
```

### 验证要点

1. **分区分配**：确认每个节点只分配了部分模型层
2. **文件过滤**：确认只下载了包含分配层的权重文件
3. **运行验证**：确认实际运行时只加载了分配的部分

### 常见问题

#### Q: 如何确认所有节点加起来覆盖了整个模型？
A: 查看分区分配日志中`total_layers_assigned`应该等于`base_shard.n_layers`。

#### Q: 为什么有些节点下载的文件比其他节点多？
A: 这是正常的，因为内存权重分配可能导致不同节点承担不同比例的模型层。

#### Q: 如何验证没有重复下载？
A: 检查每个节点的下载目录，确认没有重复的权重文件。

### 成功验证的标志

✅ 每个节点显示不同的层范围  
✅ 每个节点下载不同/数量的权重文件  
✅ 内存使用与分配的层数成正比  
✅ 所有节点加起来覆盖整个模型  
✅ 没有节点下载完整的模型权重