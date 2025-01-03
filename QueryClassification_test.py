import os
import argparse
def get_existing_files(directory):
    if not os.path.exists(directory) or not os.path.isdir(directory):
        return []  # Directory doesn't exist or is not a directory

    existing_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    files = []
    for j,i in enumerate(existing_files):
        try:
            files.append(int(i.split(".")[1].split("-")[1]))
        except:
            pass

    max(files)
    int_checkpoint = directory + "/model.ckpt-" + str(max(files))
    return int_checkpoint


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test DeepCT2 Model")
    parser.add_argument("--output_dir_train", type=str, default="DeepCT2/outputs/", help="output weights")
    parser.add_argument("--predictions_dir_test", type=str, default="DeepCT2/predictions/new_model/", help="predictions weights") 
    parser.add_argument("--max_seq_length_test", type=str, default=15, help="max_seq_length")
    parser.add_argument("--train_batch_size", type=str, default=16 , help="train_batch_size")
    parser.add_argument("--num_train_epochs", type=str, default=12, help="Number of training epochs")
    parser.add_argument("--gpu_device", type=str, default="3", help="gpu device")


    args = parser.parse_args()
    if not os.path.exists(args.predictions_dir_test):
        os.system("mkdir "+ args.predictions_dir_test)
        os.system("mkdir "+ args.predictions_dir_test)
        # os.system("mkdir "+ args.predictions_dir_test+ "pos")
        # os.system("mkdir "+ args.predictions_dir_test+ "neg")

    int_checkpoint_pos = get_existing_files(args.output_dir_train+ "pos")
    int_checkpoint_neg = get_existing_files(args.output_dir_train+ "neg")

    
    # cmd1 = "python DeepCT2/run_deepct.py " \
    # "--task_name=marcotsvdoc " \
    # "--do_train=false " \
    # "--do_eval=false " \
    # "--do_predict=true " \
    # "--data_dir=DeepCT2/data/queries.training.small.tsv " \
    # "--vocab_file=DeepCT2/uncased_L-12_H-768_A-12/vocab.txt " \
    # "--bert_config_file=DeepCT2/uncased_L-12_H-768_A-12/bert_config.json " \
    # "--init_checkpoint=" + int_checkpoint_pos + " " \
    # "--max_seq_length=" + args.max_seq_length_test + " " \
    # "--train_batch_size=" + args.train_batch_size +" " \
    # "--learning_rate=2e-5 " \
    # "--num_train_epochs=" + args.num_train_epochs +" " \
    # "--output_dir="+args.predictions_dir_test+ "pos_main " 

    # os.system(cmd1)

    # cmd2 = "python DeepCT2/run_deepct.py " \
    # "--task_name=marcotsvdoc " \
    # "--do_train=false " \
    # "--do_eval=false " \
    # "--do_predict=true " \
    # "--data_dir=DeepCT2/data/queries.training.small.tsv " \
    # "--vocab_file=DeepCT2/uncased_L-12_H-768_A-12/vocab.txt " \
    # "--bert_config_file=DeepCT2/uncased_L-12_H-768_A-12/bert_config.json " \
    # "--init_checkpoint=" + int_checkpoint_neg + " " \
    # "--max_seq_length=" + args.max_seq_length_test + " " \
    # "--train_batch_size=" + args.train_batch_size +" " \
    # "--learning_rate=2e-5 " \
    # "--num_train_epochs=" + args.num_train_epochs +" " \
    # "--output_dir="+args.predictions_dir_test+ "neg_main " 

    # os.system(cmd2)


    # years = ["dev", "2019", "2020", "2019_2020", "2021", "2022", "2021_2022", "hard"]
    years = ["dev"]#, "2019", "2020", "hard"]

    for year in years:

        cmd1 = "python DeepCT2/run_deepct.py " \
            "--task_name=marcotsvdoc " \
            "--do_train=false " \
            "--do_eval=false " \
            "--do_predict=true " \
            "--data_dir=DeepCT2/data/trec_data/" + year + "/" + year+ "_queries " \
            "--vocab_file=DeepCT2/uncased_L-12_H-768_A-12/vocab.txt " \
            "--bert_config_file=DeepCT2/uncased_L-12_H-768_A-12/bert_config.json " \
            "--init_checkpoint=" + int_checkpoint_pos + " " \
            "--max_seq_length=" + args.max_seq_length_test + " " \
            "--train_batch_size=" + args.train_batch_size +" " \
            "--learning_rate=2e-5 " \
            "--num_train_epochs=" + args.num_train_epochs +" " \
            "--output_dir="+args.predictions_dir_test+ "pos_" + year + " " 

        os.system(cmd1)
    
        cmd2 = "python DeepCT2/run_deepct.py " \
            "--task_name=marcotsvdoc " \
            "--do_train=false " \
            "--do_eval=false " \
            "--do_predict=true " \
            "--data_dir=DeepCT2/data/trec_data/" + year + "/" + year+ "_queries " \
            "--vocab_file=DeepCT2/uncased_L-12_H-768_A-12/vocab.txt " \
            "--bert_config_file=DeepCT2/uncased_L-12_H-768_A-12/bert_config.json " \
            "--init_checkpoint=" + int_checkpoint_neg + " " \
            "--max_seq_length=" + args.max_seq_length_test + " " \
            "--train_batch_size=" + args.train_batch_size +" " \
            "--learning_rate=2e-5 " \
            "--num_train_epochs=" + args.num_train_epochs +" " \
            "--output_dir="+args.predictions_dir_test+ "neg_"  + year + " " 
        
        os.system(cmd2)