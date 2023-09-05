Installation
----------

Currently, we set pytorch-lightning to version 1.9.5.
This should be done actually in torch-bsf as we don't directly use pytorch-lightning.


Files
----

- main.py:
  - 入力: 説明変数と目的変数
  - 出力: テスト誤差の計算結果と計算時間のファイル ss:paretoset ff:paretofront sf:set x front

- make_w_and_t.py:
  - 参照三角形を生成して、分割した場合はパラメータの取り直しも行う
  - 出力: 参照三角形のファイルとパラメータの取り直しが行われた三角形のファイル
    wとtがたくさん入っていて三角形の形になっている。

- jikken.py:　昔のものなので消してよい
