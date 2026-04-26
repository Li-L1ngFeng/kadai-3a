**[3a] (a) 画像検索CGI課題  画像特徴量の抽出 と 画像検索CGIの作成**
OPENCV+Pythonでカラーヒストグラムによる特徴量を事前に計算しておいて， 特徴量ファイルを読み込み画像間の距離を計算する 画像検索CGIシステムをWeb上に構築します． (Flaskに慣れている人は，CGIでなくてFlask でやってもらっても構いません．)
**よく分からない人はslack #lab_question で聞いて下さい．**

CGIを使って， [http://img.cs.uec.ac.jp/yanai/imsearch/ ](http://img.cs.uec.ac.jp/yanai/imsearch/)にあるような画像をクリックすると類似画像を類似順に表示する 画像検索システムを作ってみましょう．

画像は適当に100枚～1000枚用意してください． 画像は，[icrawler module](https://mm.cs.uec.ac.jp/local/icrawler.html) で簡単に収集できます．
以下の16種類の特徴量を使って 結果を比較しましょう．各画像のリンク先は，CGIのパラメータ埋め込みにしま す．CGIは基本はPythonですが，Perl，PHP, Rubyなどの 好きなスクリプト言語で構いません． 必ずレポートには、 数通りの検索結果画面のスクリーンキャプチャと、 システムのURLを必ず載せてください。
CGIがよく分かっていない人は [1D課題](http://jikken.cs.uec.ac.jp/1d/note.html) でネットワークの基礎を勉強してください． (ただし，言語がPerlですが．．．)
[Pythonで始めるCGIプログラム入 門](https://code-notes.com/lesson/23)， [Python CGIプログラミング入門](https://www.gesource.jp/programming/python/cgi/index.html/) なども参考にしてください．
[Flask](https://a2c.bitbucket.io/flask/)などのフレームワーク を使う手もありますが，ここではcgi moduleを使ってみましょう． (どうしてもFlaskがいい人は，flask使っても構いません．)



特徴量は事前に抽出して，特徴量ファイルを準備しておいて下さい． クエリー画像と残りの前画像との距離（もしくは類似度）を計算して， 小さい順（類似度の場合は大きい順）にソートして表示します．
以下の方法で特徴量を抽出して下さい． すべて事前に用意されているコマンドを実行するだけで出来ます． (カラーヒストグラムは簡単なので，練習に自作して下さい．難しい人は，gabor コ マンドでカラーヒストグラムも抽出できます．)

1. RGB color histogram, HSV color histogram, LUV color histogram． OpenCVで抽出して下さい．サンプルコードは [カラーヒストグラムの参考ページ](http://jikken.cs.uec.ac.jp/1d/text/4.html#1) にあります．後述する gabor コマンドにカラーヒストグラム抽出機能があるので， プログラミングしないで，それを使ってもかまいません．
2. 上記のそれぞれ 2x2の4分割，3x3の9分割． (これもgaborコマンドでできます．)これは簡単にできるでしょう．ヒストグラ ムの次元はそれぞれ4倍，9倍になります．
3. ガボール特徴量 (抽出には ~yanai/Pub/im/gaborにあるgaborコマンドを使っ てください．詳しくは同じディレクトリにあるreadme.txtを読んでください． このコマンドは color histogramも作れるのでcolor histogram作成にも利用して構い ません．) OpenCVで Gaborフィルタ使えるようです．調べてみてください．(任意 提出)
4. DCNN(Deep Convolutional Neural Network)特徴量． KerasかPyTorchでVGG16などで最終レイヤーの１つ手前の4096次元ベクトルを特 徴量として使って下さい．授業課題２－４(メディア実験も同じです)のnotebookを参考にしましょう． Wikiに書かれている様に 忘れずに l2_normを使ってL2正規化(L2ノルムが1になるようにする)をして下さい． KerasかPyTorchが自力でできない人は， [研究室Wiki](http://mm.cs.uec.ac.jp/wiki/?Deep%20Convolutional%20Feature%20%A4%CE%C3%EA%BD%D0%CA%FD%CB%A1)を参照して下さい．

カラーヒストグラムは色だけ，ガボールはテクスチャパターン， DCNNは見た目に加えて意味的な情報が，それぞれ特徴ベクトルに 含まれています．また，画像を分割するとパーツの位置が考慮された 検索結果になります．それぞれの特徴の違いを考察して下さい． なお，DCNN特徴はImageNet1000カテゴリ100万枚で学習してあるDeep Neural Network で特徴抽出していますので，1000カテゴリの意味が反映された特徴ベクトルになります．
距離は，ユークリッド(L2)距離と，ヒストグラムインターセクション(本質的に L1距離と同等）を 使って下さい．ヒストグラムインターセクションは [カラーヒストグラムの参考ページ](http://jikken.cs.uec.ac.jp/1d/text/4.html#1) に解説があります． (授業スライド：[ カラーヒストグラム(P.47-), ヒストグラムインターセクション(P.49-), ガボール(P.60-)](http://mm.cs.uec.ac.jp/yanai/report/semiobject/121026_6_ppt.pdf#page=47))
結果は，[TABLEタグ](http://www.htmq.com/html/table.shtml)で見やすく出力しましょう．（デモページの出力のHTMLを見 てみましょう．)
なお，CGIは，自分のホームに www というディレクトリを作って， chmod a+rx www としてから 以下のように書いたtext fileを

```
Options ExecCGI FollowSymLinks
order deny,allow
deny from all
allow from 130.153.192.
satisfy any

AuthUserFile "/home/yanai-lab/{User ID}/.htpw"
AuthName "Enter password"
AuthType basic
require valid-user
```

.htaccess という名前で用意して，wwwディレクトリに置いてください． その下に適当にディレクトリを作ってCGIを置いてください．(自宅からアクセス したい場合は，mm.cs.uec.ac.jpのhtpasswdを設定してください． その場合は，必ず，ID:yanailab, PW:w9-707 を入れてください．
ssh mm htpasswd -c -b ~/.htpw yanailab w9-707 で設定できます．(-c はPWファイル初期化なので，2つ目以降のPW登録には付けないこと．) )
CGIファイルは，chmod 755 index.cgi などとして，実行可能ファイルにして下さい． さらに，コマンドラインから，./index.cgi として実行できるか 必ず確認して下さい．コマンドラインから実行できない場合は，ブラウザからも 当然，実行できません．
Webサーバは，mm.cs.uec.ac.jp, img.cs.uec.ac.jp の2台あります．imgは旧環境なのでpython2しか入っていませんので， 原則 mm を使ってください． mm は(GPUは非搭載ですが)，/usr/local/anaconda3/bin/python3 を使えば，cgiモジュールはもちろん，pytorchやKerasも使えます． ただし，mmの場合は外部に公開されてしまうので，上記のようにアクセス制限を必ず設定して下さい．
例えば，search.cgi を ~/www/imsearch/ に作れば， http://mm.cs.uec.ac.jp/(アカウント名)/imsearch/search.cgi でアクセスできます． また，CGIの名前を index.cgi とすれば，cgi ファイル名は省略できて， http://mm.cs.uec.ac.jp/(アカウント名)/imsearch/ でアクセスできます． **レポートには必ず作ったサイトのURLを書いて下さい．**
なお，画像データは wwwの下には直接置かないで， /export/space0/ に自分のユーザ名でディレクトリを作ってそこに置いてくださ い．例えば，~/www/imsearch/ から，ln -s /export/space0/(自分のアカウント名)/imgdata などとして，シンボリックリンクを張って利用して下さい．
**詳しくは回りの院生に聞いて下さい．**

**各自のhomeには画像データは原則置かないことになっています．**homeはバッ クアップを定期的にバックアップを取っていますが，容量が 20TBしかありません ので節約するためです．/export/space0 は 430TBあります．バックアップは取っ ていませんが，RAID6+分散ファイルシステム なので データは厳重に守られています．ただし，バックアッ プがないので，間違って rm すると消えてしまいます．一方，homeは間違って rmしても週1回金曜日の深夜にバックアップをしているので，直前のバックアッ プ状態までは復活出来ます．復活が必要な場合は，柳井までメール下さい．