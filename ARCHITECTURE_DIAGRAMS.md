# Exo 系统架构图文档

本文档通过 PlantUML 图表展示了 Exo 分布式 AI 推理系统的主要类结构和对象关系。

## 图表概览

### 颜色编码说明
所有图表都使用统一的颜色编码系统来区分不同类型的关系：

- **🔴 红色 (#FF6B6B)**: 用户交互和创建关系
- **🟢 青色 (#4ECDC4)**: 系统管理和组合关系
- **🔵 蓝色 (#45B7D1)**: 继承和实现关系
- **🟢 绿色 (#96CEB4)**: 网络通信和数据流
- **🟡 黄色 (#FFEAA7)**: 引擎操作和推理流程
- **🟣 紫色 (#DDA0DD)**: 回调系统和异步通信
- **🟠 橙色 (#F8C471)**: 数据管理和映射关系

### 1. 核心对象图 (`exo_architecture_object_diagram.puml`)
展示系统中主要对象实例及其关系：
- **Node**: 中央协调器
- **Topology**: 网络状态管理
- **Discovery**: 服务发现（UDP/Tailscale/Manual）
- **InferenceEngine**: 推理引擎（MLX/TinyGrad）
- **Shard**: 模型分片
- **ChatGPTAPI**: REST API接口

**颜色区分**：
- 红色线条：对象创建关系
- 青色线条：组合关系(has-a)
- 蓝色线条：继承关系
- 绿色线条：网络通信
- 紫色线条：回调系统

### 2. 详细类图 (`exo_detailed_class_diagram.puml`)
包含完整的类继承关系和关键方法：
- 抽象基类：InferenceEngine, Discovery, PeerHandle
- 具体实现类：MLXDynamicShardInferenceEngine, TinygradDynamicShardInferenceEngine
- 网络层：GRPCServer, 各种Discovery实现
- 数据对象：Shard, Topology, Partition, DeviceCapabilities

**颜色区分**：
- 蓝色背景：抽象类
- 绿色背景：具体实现类
- 蓝色线条：继承关系
- 青色线条：组合关系
- 黄色线条：使用关系

### 3. 核心流程图 (`exo_core_flow_diagram.puml`)
展示对象间的交互消息流：
- 系统启动流程
- 模型加载流程
- 分布式推理流程
- 状态更新机制

**颜色区分**：
- 红色：用户交互
- 青色：系统管理
- 蓝色：数据流
- 绿色：网络通信
- 紫色：异步回调

### 4. 组件架构图 (`exo_component_architecture.puml`)
分层架构视角：
- 应用层：main.py, API, CLI
- 编排层：Node, Topology管理
- 网络层：GRPC Server, Discovery服务
- 推理层：各种Inference Engine实现
- 模型管理层：Shard下载和管理
- 可视化层：Topology可视化

**颜色区分**：
- 红色：应用层依赖
- 青色：编排层管理
- 蓝色：网络层通信
- 绿色：推理层操作
- 黄色：模型管理
- 紫色：可视化更新

### 5. 分布式推理序列图 (`exo_distributed_inference_sequence.puml`)
完整的分布式推理流程：
- 用户请求处理
- 网络拓扑分析
- 模型分片加载
- 多节点并行处理
- 结果聚合返回

**颜色区分**：
- 红色：用户请求
- 青色：API处理
- 蓝色：网络分析
- 黄色：引擎推理
- 绿色：节点通信
- 紫色：模型下载

## 关键设计模式

### 1. 抽象工厂模式
- `InferenceEngine` 作为抽象基类，支持多种后端实现
- `Discovery` 抽象服务发现机制

### 2. 策略模式
- `PartitioningStrategy` 支持不同的分片策略
- `RingMemoryWeightedPartitioningStrategy` 基于内存权重的实现

### 3. 观察者模式
- Node 的回调系统：on_token, on_opaque_status
- 异步事件处理机制

### 4. 代理模式
- `PeerHandle` 抽象远程节点通信
- `GRPCPeerHandle` 实现基于GRPC的通信

## 核心对象关系

### Node 作为中央协调器
```
Node 组合关系：
- has-a Topology (网络状态)
- has-a InferenceEngine (推理引擎)
- has-a Discovery (服务发现)
- has-a Server (网络接口)
- has-a ShardDownloader (模型管理)
```

### 分布式推理流程
```
1. 用户请求 → ChatGPTAPI → Node
2. Node 分析拓扑 → 生成分区 → 分配任务
3. 本地推理 + 远程节点协作
4. 结果聚合 → 返回给用户
```

## 使用方法

1. 安装 PlantUML 插件或使用在线渲染器
2. 打开 `.puml` 文件查看图表
3. 可以修改图表来适应新的需求

## 扩展建议

添加新功能时，可以：
1. 创建新的 InferenceEngine 实现
2. 添加新的 Discovery 机制
3. 实现新的 PartitioningStrategy
4. 扩展 PeerHandle 协议

这些图表提供了系统的完整架构视图，有助于理解对象间的复杂关系和设计决策。