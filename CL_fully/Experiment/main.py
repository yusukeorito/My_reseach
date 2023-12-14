import argparse
import yaml
import sys
import logging

# loggerオブジェクトの作成
logger = logging.getLogger("ログ太郎")
# ログレベルを設定
logger.setLevel(logging.DEBUG)  # ログレベルを DEBUG に設定

# ハンドラの作成
console_handler = logging.StreamHandler()  # コンソールに出力するハンドラを作成
logger.addHandler(console_handler)  # ハンドラをルートロガーに追加

sys.path.append('./')
from runner import Runner

if __name__ == "__main__":
    """
    cd experiments
    python main.py --run ../Configs/run001.yml
    """
    parser = argparse.ArgumentParser()#オブジェクト生成
    parser.add_argument("--run")#引数の追加
    args = parser.parse_args()
    
    if args.run:
        #--run オプションが実行されるとargs.runにパスが格納され条件分が真になる
        f = open(args.run, "r+")
        run_configs = yaml.load(f, Loader=yaml.SafeLoader)

        
        runner = Runner(run_configs)
        X_train,y_train, X_test, y_test = runner.preprocessing()
        X_train,X_test, pca1, scaler1 = runner.PCA_SS(X_train, X_test)
        runner.train_model(X_train, y_train, X_test, y_test)
        

                