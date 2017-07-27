# Filter_smoothing
C言語による粒子フィルタの実装
計算時間克服のため



ブランチごとのSourceの内容

- master
- challenge_plot  
guniplotの練習用
- check_rnorm  
正規分布乱数が想定したものなっているか確認
- check_time  
時間計測用
- dynamic_default  
DynamicDefaultRatesモデル (Lamb and Perraudinモデル)でのEMアルゴリズムの実装  
- multisred  
マルチスレッドに挑戦
- dynamic_default_c++  
C++に適用済み、他とプロジェクト名から違うため注意 2017/07/27現在の作業ブランチ
- dynamic_default_c++_fitdata    
C++でデータ分析　米国のデフォルトデータ
- dd_c++_check  
c++で作成した各関数が、想定通り動いている確認するためのブランチ
- check_grad  
c++で計算した勾配を確認
