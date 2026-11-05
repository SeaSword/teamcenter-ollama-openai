# Upstream workflow references / 上游工作流参考

These files preserve the original project's CI and publishing definitions as reference source. GitHub does not execute workflows from this directory. Their package names, registries, credentials, and release policies belong to the upstream project and must be reviewed before any reuse.

此目录保留上游 CI 与发布流程源码，仅供参考。GitHub 不会从此目录执行工作流。文件中的包名、镜像仓库、凭据名称及发布策略属于上游，复用前需重新配置。

This repository's active workflow is `../workflows/compatibility.yml`, which runs mocked protocol tests without publishing packages or images.

本仓库活动流程为 `../workflows/compatibility.yml`，只进行 Mock 协议测试，不自动发布包或镜像。
