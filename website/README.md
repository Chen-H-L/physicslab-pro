# 物演智启网站部署说明

这个目录是项目官网的静态站点，可以直接部署到 GitHub Pages 或 Cloudflare Pages。

## 下载按钮

网站中的“获取程序 / 下载 Windows 版 / 下载程序”按钮统一指向：

```text
https://github.com/Chen-H-L/physicslab-pro/releases/latest/download/PhysicsLabPro.zip
```

新版 EXE 打包完成后，不要把 `PhysicsLabPro.zip` 放进网站目录提交到仓库。应该在 GitHub 仓库的 Releases 页面创建新版 Release，并上传压缩包，文件名保持为 `PhysicsLabPro.zip`。

## GitHub Pages

仓库已经提供 `.github/workflows/pages.yml`。推送到 `main` 分支后，GitHub Actions 会把 `website` 目录作为静态站点发布。

发布前需要在 GitHub 仓库中启用 Pages：

1. 打开仓库 Settings。
2. 进入 Pages。
3. Source 选择 GitHub Actions。
4. 推送代码后等待 Actions 完成。

发布地址通常是：

```text
https://chen-h-l.github.io/physicslab-pro/
```

## Cloudflare Pages

如果使用 Cloudflare Pages：

- Build command 留空。
- Build output directory 填写 `website`。
- 不要上传 `website/downloads` 中的大文件。

程序压缩包继续通过 GitHub Releases 提供下载。
