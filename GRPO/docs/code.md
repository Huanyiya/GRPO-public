文件	核心作用	使用场景
dapo_math.py	提取数学答案，与 Ground Truth 比较，正确返回 1，否则返回 0	Math GRPO
acecode.py	提取模型生成的 Python 代码，运行 AceCode 测试用例，全部通过返回 1，否则返回 0	AceCode GRPO
code.py	处理 Code 输出、提取 Python 代码，支持标准输入输出和函数调用形式的测试	Eurus Code，以及 AceCode 的代码提取
code_compare.py	比较程序运行结果与预期输出，支持空白字符规范化、数值容差等	主要供 code.py 使用
code_sandbox.py	通过 HTTP 请求 SandboxFusion 远程执行代码，管理超时、并发及执行结果	AceCode / Eurus Code
__init__.py	声明 rewards 是 Python 包，目前只有一行说明，没有额外逻辑	模块导入

模型生成 Python 代码，执行测试用例，全部通过才获得 Reward = 1，否则 Reward = 0。


时间类别	计时范围	包含的主要操作
Rollout	开始采样 → 收集完所有轨迹	Student 生成 Token；Agent 场景还包括环境交互、工具调用、等待执行结果
Advantages	收集完 Reward → 计算完 Advantage	Reward 处理、GRPO 组内 Reward 均值/标准差、Advantage 计算
Policy/ref scoring	开始重算策略概率 → 得到所需 Log-probs	当前 Policy 对轨迹做 Teacher Forcing，以及可选 Reference Policy 的前向打分
Policy update	开始策略优化 → Optimizer Step 完成	Loss 计算、Backward、梯度同步、梯度裁剪、Optimizer Step
Teacher forward	开始蒸馏 Teacher 前向 → 完成 Teacher 打分	Teacher 对 Student 轨迹进行前向推理，取得 Logits、Log-probs 或其他蒸馏信号
Other	以上阶段之外的时间	数据准备、进程切换、权重同步、等待、Checkpoint 保存及其他框架开销