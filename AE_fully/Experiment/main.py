import argparse
import yaml
import sys
import logging
import time

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
    start_time = time.time()
    parser = argparse.ArgumentParser()#オブジェクト生成
    parser.add_argument("--run")#引数の追加
    args = parser.parse_args()
    
    if args.run:
        #--run オプションが実行されるとargs.runにパスが格納され条件分が真になる
        f = open(args.run, "r+")
        run_configs = yaml.load(f, Loader=yaml.SafeLoader)
        runner = Runner(run_configs)
        X_train,y_train, X_test, y_test = runner.preprocessing()
        
        train_in,train_out, pca1, scaler1 = PCA_SS_func(train_in, train_out)
        test_in,test_out, pca2, scaler2 = PCA_SS_func(test_in, test_out)
        
        runner.train_model(train_in, train_out, test_in, test_out)
      
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"実行にかかった時間: {execution_time}秒")
