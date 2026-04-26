# 画像検索 CGI 実装メモ

## 現在の実装範囲
- 色ヒストグラム: rgb, hsv, luv
- 空間分割: 2x2, 3x3
- テクスチャ: gabor (OpenCV)
- 距離/類似度: l2, intersection
- CGI 検索ページ: `cgi/index.cgi`

## 一键构建索引
```bash
cd /export/space0/li-l/kadai3a
chmod +x scripts/build_index_all.sh
scripts/build_index_all.sh /export/space0/li-l/imgdata/kadai3a /export/space0/li-l/kadai3a/features
```

说明:
- 当前脚本优先使用 PyTorch 提取 DCNN（VGG16 fc2=4096）；若不可用会自动跳过。
- 可用于 CGI 的索引:
  - 至少: `features/index_plus_gabor.npz`
  - 若 DCNN 成功: `features/index_full.npz`

## 手动检索测试
```bash
cd /export/space0/li-l/kadai3a
/usr/local/anaconda3/bin/conda run -p /home/yanai-lab/li-l/.conda/envs/env1 --no-capture-output python /host/home/yanai-lab/Sotsuken25/li-l/.vscode-server/extensions/ms-python.python-2026.4.0-linux-x64/python_files/get_output_via_markers.py scripts/search_core.py --index features/index_plus_gabor.npz --query /export/space0/li-l/imgdata/kadai3a/000001.jpg --feature gabor --metric l2 --topk 10
```

## CGI 命令行测试
```bash
cd /export/space0/li-l/kadai3a
FEATURES_PATH=/export/space0/li-l/kadai3a/features/index_plus_gabor.npz \
IMG_ROOT=/export/space0/li-l/imgdata/kadai3a \
IMG_BASE_URL=imgdata/kadai3a \
QUERY_STRING='query=/export/space0/li-l/imgdata/kadai3a/000001.jpg&feature=gabor&metric=l2&topk=10' \
./cgi/index.cgi

# 若已生成 index_full.npz，可将 gabor 改为 dcnn 验证
# FEATURES_PATH=/export/space0/li-l/kadai3a/features/index_full.npz
# QUERY_STRING='query=/export/space0/li-l/imgdata/kadai3a/000001.jpg&feature=dcnn&metric=l2&topk=10'
```

## mm 服务器部署
1. 将 `cgi/index.cgi` 复制到 `~/www/imsearch/index.cgi` 并 `chmod 755`。
2. 将 `cgi/.htaccess.sample` 复制为 `~/www/.htaccess` 并按账号路径修改 `AuthUserFile`。
3. 确保 `~/www/imsearch` 下可访问图片目录软链接 `imgdata -> /export/space0/li-l/imgdata`。
4. 用浏览器访问 `http://mm.cs.uec.ac.jp/li-l/imsearch/` 验证。