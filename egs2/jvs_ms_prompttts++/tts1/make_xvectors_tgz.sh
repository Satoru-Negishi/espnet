#!/bin/bash

# シェルスクリプトの実行ディレクトリを変数に格納
base_dir=$(pwd)

# アーカイブ対象のディレクトリを指定
source_dir="dump/22k/xvector/"

# 一時ディレクトリを作成してアーカイブ対象のファイルをコピー
temp_dir=$(mktemp -d)
for target_dir in eval1 dev tr_no_dev; do
    mkdir -p "$temp_dir/$source_dir/$target_dir"
    cp -r "$base_dir/$source_dir/$target_dir"/spk_xvector.ark "$base_dir/$source_dir/$target_dir"/spk_xvector.scp "$temp_dir/$source_dir/$target_dir"
done

# アーカイブファイルを作成
tar czf "$base_dir/xvectors.tgz" -C "$temp_dir" .

# 一時ディレクトリを削除
rm -rf "$temp_dir"

echo "xvectors.tgz ファイルの作成が完了しました。"
