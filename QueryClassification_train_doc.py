import os
import subprocess
import argparse

if __name__ == "__main__":

    # cudnn_path = os.path.dirname(subprocess.check_output(["python", "-c", "import nvidia.cudnn;print(nvidia.cudnn.__file__)"], universal_newlines=True).strip())
    # os.environ['LD_LIBRARY_PATH'] = f"{cudnn_path}/lib:{os.environ['CONDA_PREFIX']}/lib/:{os.environ['LD_LIBRARY_PATH']}"

    # Argument parser
    parser = argparse.ArgumentParser(description="Train and Test DeepCT2 Model")
    parser.add_argument("--data_dir_train", type=str, default="DeepCT2/data/", help="directory of neg or pos train data")
    parser.add_argument("--label_method", type=str, default="QueryMethod", help="directory of neg or pos train data")
    parser.add_argument("--max_seq_length_train", type=str, default=9, help="max_seq_length")
    parser.add_argument("--train_batch_size", type=str, default=16 , help="output path")
    parser.add_argument("--num_train_epochs", type=str, default=12, help="Number of training epochs")
    parser.add_argument("--output_dir_train", type=str, default="DeepCT2/outputs/", help="output weights")
    parser.add_argument("--gpu_device", type=str, default="3", help="gpu device")
    parser.add_argument("--alpha", type=str, default="0.15", help="difficulty lower bound")

    args = parser.parse_args()
    if not os.path.exists(args.output_dir_train + "pos"):
        os.system("mkdir "+ args.output_dir_train+ "pos")
        os.system("mkdir "+ args.output_dir_train+ "neg")
# args.output + "_pos_"+ args.method + "_" + str(args.alpha)[0:4] +".json"
    # for negative labels
    cmd1 = "python DeepCT2/run_deepct.py " \
    "--task_name=marcodoc " \
    "--do_train=true " \
    "--do_eval=false " \
    "--do_predict=false " \
    "--data_dir="+ args.data_dir_train + "_neg_"+ args.label_method + "_" + args.alpha + ".json "  \
    "--vocab_file=DeepCT2/uncased_L-12_H-768_A-12/vocab.txt " \
    "--bert_config_file=DeepCT2/uncased_L-12_H-768_A-12/bert_config.json " \
    "--init_checkpoint=DeepCT2/uncased_L-12_H-768_A-12/bert_model.ckpt " \
    "--max_seq_length=" + args.max_seq_length_train + " " \
    "--train_batch_size=" + args.train_batch_size + " " \
    "--learning_rate=2e-5 " \
    "--num_train_epochs=" + args.num_train_epochs  + " "\
    "--recall_field=title " \
    "--gpu_device=" + args.gpu_device + " " \
    "--output_dir=" + args.output_dir_train + "neg "

    os.system(cmd1)

    # for positive labels

    cmd1 = "python DeepCT2/run_deepct.py " \
    "--task_name=marcodoc " \
    "--do_train=true " \
    "--do_eval=false " \
    "--do_predict=false " \
    "--data_dir="+ args.data_dir_train + "_pos_"+ args.label_method + "_" + args.alpha + ".json "  \
    "--vocab_file=DeepCT2/uncased_L-12_H-768_A-12/vocab.txt " \
    "--bert_config_file=DeepCT2/uncased_L-12_H-768_A-12/bert_config.json " \
    "--init_checkpoint=DeepCT2/uncased_L-12_H-768_A-12/bert_model.ckpt " \
    "--max_seq_length=" + args.max_seq_length_train + " " \
    "--train_batch_size=" + args.train_batch_size + " " \
    "--learning_rate=2e-5 " \
    "--num_train_epochs=" + args.num_train_epochs  + " "\
    "--recall_field=title " \
    "--gpu_device=" + args.gpu_device + " " \
    "--output_dir=" + args.output_dir_train + "pos "

    os.system(cmd1)
