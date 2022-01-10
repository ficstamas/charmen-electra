from model.electra import Electra
from model.finetuning.charformer import ElectraCharformer
from model.finetuning.electra import Electra
from argparse import ArgumentParser
import pandas as pd
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
from sklearn.metrics import classification_report


def main():
    ckp = args.checkpoint
    if args.charformer:
        model = ElectraCharformer(data_path=args.data)
    else:
        model = Electra(data_path=args.data)
    trainer = model.make_model(ckp, args.local_rank, batch_size=args.batch_size, output=args.output,
                               max_block_size=args.max_block_size, downsample_factor=args.downsample_factor,
                               max_length=args.max_length, score_consensus_attn=args.score_consensus_attn,
                               upsample_output=args.upsample_output, binary=args.binary, num_epochs=args.num_epochs,
                               lr=args.lr, freeze=args.freeze, weight_decay=args.weight_decay, adam_eps=args.adam_eps)

    trainer.train(resume_from_checkpoint=False)

    df = model.evaluate(trainer)
    df.to_csv(os.path.join(args.output, "predictions.csv"))
    print("Validation")
    print_summary(model.validation(trainer))
    print("Test")
    print_summary(df)


def print_summary(df: pd.DataFrame):
    logging.info(
        classification_report(df["Labels"], df["Predictions"], labels=[0, 1, 2] if not args.binary else [0, 1])
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", default="model-cps/charformer/block-4_ds-4_seq-1024/pytorch_model.bin")
    parser.add_argument("--output", default="test/")
    parser.add_argument("--data", default="model-cps/OpinHuBank_20130106_updated.xls")
    parser.add_argument("--charformer", default=False, action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--max_block_size", type=int, default=4)
    parser.add_argument("--downsample_factor", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--freeze", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--score_consensus_attn", default=False, action="store_true")
    parser.add_argument("--upsample_output", default=False, action="store_true")
    parser.add_argument("--binary", default=False, action="store_true")
    args = parser.parse_args()
    main()

