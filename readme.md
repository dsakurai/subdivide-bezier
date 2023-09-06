Installation
----------

Currently, we set pytorch-lightning to version 1.9.5.
This should be done actually in torch-bsf as we don't directly use pytorch-lightning.


Files
----

- core.py:
  - 入力: 説明変数と目的変数
  - 出力: テスト誤差の計算結果と計算時間
    - You can run tests in this Python file
        - The errors between the Pareto front/set and their approximation are expected to be the same as the values hard-coded in the test
        - In contrast, the time for training may vary depending on the hardware
