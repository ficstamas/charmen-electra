from model.electra import Electra
from model.electra_charformer import ElectraCharformer
from argparse import ArgumentParser
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def main():
    # /data2/ficstamas/webcorpus-hu-2.0/
    # /data2/ficstamas/MILab/electra_hubert-token/trainer-experiments/

    ckp = args.checkpoint
    if args.charformer:
        model = ElectraCharformer(data_path=args.data)
    else:
        model = Electra(data_path=args.data, data_split=args.split)
    trainer = model.make_model(ckp, args.local_rank, batch_size=args.batch_size)

    if args.resume:
        trainer.train(resume_from_checkpoint=os.path.exists(ckp))
    else:
        trainer.train(resume_from_checkpoint=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", default="/data2/ficstamas/MILab/electra_hubert-token/trainer-experiments/")
    parser.add_argument("--data", default="/data2/ficstamas/webcorpus-hu-2.0/")
    parser.add_argument("--charformer", default=False, action="store_true")
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    main()
