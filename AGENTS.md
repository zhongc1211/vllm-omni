# AGENTS.md

本文件是在本仓库（zhongc1211/vllm-omni 及其 worktree）里干活的规矩。与派活说明冲突时，以派活说明里明确写出的为准；其他情况以本文件为准。

## 开工前先查 EverOS 经验

- 动手前调用 EverOS `recall_agent_experience`（task 写几个关键词：模块名、文件名、报错原文，比如 `NCCL`），冲突时以本文件为准；不要调写入类 everos 工具。把命中的经验贴进计划或 brief；没命中也写「EverOS recall：无命中，查询词 …」。
  - Claude Code 在本仓库的目录里会自动挂上 everos（只开 `recall_agent_experience`、`search_memory`、`list_memories` 三个只读工具）。覆盖范围：box 上的 `/workspace/repos/vllm-omni` 和 `~/orca/workspaces/vllm-omni/*`，iMac 上 `~/orca/workspaces/repository/*`（bare 仓库 `~/pi/remote/h3-c1-lab/repository.git` 的 worktree）。iMac 上的 `~/pi/remote/vllm-omni`、`~/claude/gpu/vllm-omni` 不挂。
  - 自己开 worker 时关掉：Claude 加 `--disallowedTools mcp__everos`，pi 加 `--exclude-tools 'mcp__everos__*'`；worker 只读父会话贴进 brief 的 recall。key 不打印、不进仓库。
  - 作用域：`app_id=vllm-omni`、`project_id=vllm-omni`、`agent_id=coding-vllm-omni`、`user_id=zack`（按 app_id + project_id 隔离）。recall 不可用时，在报告里写明 recall 没跑及原因，再照常按派活说明执行。
